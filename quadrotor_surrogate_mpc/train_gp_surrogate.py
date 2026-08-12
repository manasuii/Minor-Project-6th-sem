"""
train_gp_surrogate.py
=====================
Trains 6 independent Sparse Variational Gaussian Process surrogates (one per
output channel) using GPyTorch with inducing-point approximation (SVGP).

Each GP learns the SAME physics-informed target as the NN surrogate — the
nonlinear residual (next_state - state) - delta_linear(state, input) — so that
the open-loop three-way comparison (O2) is strictly apples-to-apples.

Uses a 10,000-sample subset of the training split (GPs do not scale to the
full set as cheaply as NNs, and 10k gives good accuracy here).

Saves:
  models/gp_surrogate_{i}.pth  — model + likelihood state for channel i (0..5)
  models/gp_cfg.npz            — input + residual normalisation stats, n_inducing
"""

import os
import time

import numpy as np
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

from surrogate_common import (
    SVGPModel, DATA_DIR, DEVICE,
    load_splits, load_subset, input_norm_stats, residual_norm_stats,
    nonlinear_residual,
)

# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
N_GP_TRAIN    = 10_000
N_INDUCING    = 500
EPOCHS_GP     = 100
BATCH_SIZE_GP = 512
LR_GP         = 0.01
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)


# ─────────────────────────────────────────
#  Data
# ─────────────────────────────────────────
def load_gp_data(n_train=N_GP_TRAIN, data_dir=DATA_DIR):
    """Residual-target training subset + full test set, normalised."""
    s = load_splits(data_dir)
    X_mean, X_std = input_norm_stats(s["X_train"])
    Y_mean, Y_std = residual_norm_stats(s["X_train"], s["Y_train"])

    Xsub, Ysub = load_subset(n_train, data_dir, seed=RANDOM_SEED)
    Xtr = (Xsub.astype(np.float32) - X_mean) / X_std
    Ytr = (nonlinear_residual(Xsub, Ysub).astype(np.float32) - Y_mean) / Y_std

    Xte = (s["X_test"].astype(np.float32) - X_mean) / X_std
    Yte_raw = nonlinear_residual(s["X_test"], s["Y_test"]).astype(np.float32)

    return (torch.tensor(Xtr), torch.tensor(Ytr),
            torch.tensor(Xte), Yte_raw,
            X_mean, X_std, Y_mean, Y_std)


# ─────────────────────────────────────────
#  Train one GP per channel
# ─────────────────────────────────────────
def train_one_gp(channel_idx, Xtr, Ytr, epochs=EPOCHS_GP, n_inducing=N_INDUCING,
                 verbose=True):
    y = Ytr[:, channel_idx]
    ind_idx = torch.randperm(len(Xtr))[:n_inducing]
    inducing = Xtr[ind_idx].clone().to(DEVICE)

    model = SVGPModel(inducing).to(DEVICE)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
    model.train(); lik.train()

    opt = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": lik.parameters()}], lr=LR_GP
    )
    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=len(Xtr))
    loader = DataLoader(TensorDataset(Xtr.to(DEVICE), y.to(DEVICE)),
                        batch_size=BATCH_SIZE_GP, shuffle=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            opt.step()
        if verbose and epoch % 25 == 0:
            print(f"    Channel {channel_idx}  Epoch {epoch:3d}  ELBO = {-loss.item():.4f}")
    return model, lik


def train_gp(X, Y, epochs=EPOCHS_GP, n_inducing=N_INDUCING, seed=RANDOM_SEED,
             verbose=False):
    """
    Reusable trainer for the sample-efficiency sweep.
    Returns (models, likelihoods, cfg-dict).
    """
    torch.manual_seed(seed)
    X_mean, X_std = input_norm_stats(X)
    Y_mean, Y_std = residual_norm_stats(X, Y)
    Xtr = torch.tensor((X.astype(np.float32) - X_mean) / X_std)
    Ytr = torch.tensor((nonlinear_residual(X, Y).astype(np.float32) - Y_mean) / Y_std)

    n_ind = min(n_inducing, len(Xtr))
    models, liks = [], []
    for ch in range(6):
        m, lk = train_one_gp(ch, Xtr, Ytr, epochs=epochs, n_inducing=n_ind, verbose=verbose)
        models.append(m.eval()); liks.append(lk.eval())
    cfg = {"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std,
           "n_inducing": n_ind}
    return models, liks, cfg


# ─────────────────────────────────────────
#  Evaluate
# ─────────────────────────────────────────
def evaluate_gp(models, liks, Xte, Yte_raw, Y_mean, Y_std):
    names = ["d_phi", "d_theta", "d_psi", "d_p", "d_q", "d_r"]
    print("\n── GP Test-set nonlinear-residual RMSE (physical units) ──")
    preds = []
    for i, (m, lk) in enumerate(zip(models, liks)):
        m.eval(); lk.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist = lk(m(Xte.to(DEVICE)))
            mean = dist.mean.cpu().numpy(); std = dist.stddev.cpu().numpy()
        phys = mean * Y_std[i] + Y_mean[i]
        rmse = np.sqrt(np.mean((phys - Yte_raw[:, i]) ** 2))
        print(f"  {names[i]:9s}: RMSE = {rmse:.2e}  |  mean uncertainty = "
              f"{std.mean() * Y_std[i]:.2e}")
        preds.append(phys)
    return np.stack(preds, axis=1)


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Training GP Surrogate for Quadrotor Attitude Dynamics")
    print(f"Using {N_GP_TRAIN:,} training points, {N_INDUCING} inducing points")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    Xtr, Ytr, Xte, Yte_raw, X_mean, X_std, Y_mean, Y_std = load_gp_data()
    print(f"GP training set: {len(Xtr):,} samples")

    models, liks = [], []
    t0 = time.time()
    for ch in range(6):
        print(f"\nTraining GP for channel {ch} ...")
        m, lk = train_one_gp(ch, Xtr, Ytr)
        torch.save({"model": m.state_dict(), "likelihood": lk.state_dict()},
                   f"models/gp_surrogate_{ch}.pth")
        models.append(m); liks.append(lk)
    print(f"\nTotal GP training time: {time.time() - t0:.1f} s")

    np.savez("models/gp_cfg.npz",
             X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
             n_inducing=N_INDUCING)

    evaluate_gp(models, liks, Xte, Yte_raw, Y_mean, Y_std)
    print("\n✓ GP Surrogate training complete")
    print("  Saved: models/gp_surrogate_0.pth ... models/gp_surrogate_5.pth")
