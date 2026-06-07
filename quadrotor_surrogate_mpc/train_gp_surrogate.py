"""
train_gp_surrogate.py
=====================
Trains 6 independent sparse Gaussian Process surrogates (one per output
channel) using GPyTorch with inducing-point approximation (SVGP).

Uses 10,000 training samples (subset of full dataset — GPs don't scale
as well as NNs, but 10k is sufficient for good accuracy here).

Saves:
  models/gp_surrogate_{i}.pth  — model state for channel i (i=0..5)
  models/gp_cfg.npz            — normalisation stats
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.utils.data import TensorDataset, DataLoader
import os
import time


# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
N_GP_TRAIN    = 10_000   # GP subset (full 35k is too slow)
N_INDUCING    = 500
EPOCHS_GP     = 100
BATCH_SIZE_GP = 512
LR_GP         = 0.01
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────
#  Sparse GP model (SVGP)
# ─────────────────────────────────────────
class SVGPModel(ApproximateGP):
    """Sparse Variational Gaussian Process with RBF + Matern kernel."""

    def __init__(self, inducing_points):
        variational_dist = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_dist, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        mean  = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ─────────────────────────────────────────
#  Load and prepare data
# ─────────────────────────────────────────
def load_gp_data(n_train=N_GP_TRAIN):
    """Load a subset of the training data for GP fitting."""
    train = np.load("data/train.npz")
    stats = np.load("data/normstats.npz")
    test  = np.load("data/test.npz")

    X_mean = stats["X_mean"].astype(np.float32)
    X_std  = stats["X_std"].astype(np.float32)
    Y_mean = stats["Y_mean"].astype(np.float32)
    Y_std  = stats["Y_std"].astype(np.float32)

    # Subset for training
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(train["X"]), size=n_train, replace=False)

    Xtr = (train["X"][idx].astype(np.float32) - X_mean) / X_std
    Ytr = (train["Y"][idx].astype(np.float32) - Y_mean) / Y_std

    Xte = (test["X"].astype(np.float32) - X_mean) / X_std
    Yte = test["Y"].astype(np.float32)

    return (torch.tensor(Xtr), torch.tensor(Ytr),
            torch.tensor(Xte), Yte,
            X_mean, X_std, Y_mean, Y_std)


# ─────────────────────────────────────────
#  Train one GP per output channel
# ─────────────────────────────────────────
def train_one_gp(channel_idx, Xtr, Ytr, epochs=EPOCHS_GP):
    """Train SVGP for a single output channel."""
    y_channel = Ytr[:, channel_idx]

    # Select inducing points from training data
    ind_idx = torch.randperm(len(Xtr))[:N_INDUCING]
    inducing_points = Xtr[ind_idx].clone().to(DEVICE)

    model      = SVGPModel(inducing_points).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    model.train()
    likelihood.train()

    optimiser = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=LR_GP)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(Xtr))
    dataset = TensorDataset(Xtr.to(DEVICE), y_channel.to(DEVICE))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE_GP, shuffle=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            optimiser.zero_grad()
            output = model(xb)
            loss = -mll(output, yb)
            loss.backward()
            optimiser.step()

        if epoch % 25 == 0:
            print(f"    Channel {channel_idx}  Epoch {epoch:3d}  ELBO = {-loss.item():.4f}")

    return model, likelihood


# ─────────────────────────────────────────
#  Evaluate GP
# ─────────────────────────────────────────
def evaluate_gp(models, likelihoods, Xte, Yte_raw, Y_mean, Y_std):
    """Evaluate all 6 GPs on the test set."""
    channel_names = ["Δphi", "Δtheta", "Δpsi", "Δp", "Δq", "Δr"]
    print("\n── GP Test Set RMSE (physical units) ──")
    all_preds = []

    for i, (model, likelihood) in enumerate(zip(models, likelihoods)):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = likelihood(model(Xte.to(DEVICE)))
            pred_mean = pred_dist.mean.cpu().numpy()
            pred_std  = pred_dist.stddev.cpu().numpy()

        # Denormalise
        pred_phys = pred_mean * Y_std[i] + Y_mean[i]
        rmse = np.sqrt(np.mean((pred_phys - Yte_raw[:, i])**2))
        print(f"  {channel_names[i]:8s}: RMSE = {rmse:.2e}  |  "
              f"mean uncertainty = {pred_std.mean() * Y_std[i]:.2e}")
        all_preds.append(pred_phys)

    return np.stack(all_preds, axis=1)


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

    models, likelihoods = [], []
    t0 = time.time()

    for ch in range(6):
        print(f"\nTraining GP for channel {ch} ...")
        m, l = train_one_gp(ch, Xtr, Ytr)
        torch.save({"model": m.state_dict(), "likelihood": l.state_dict()},
                   f"models/gp_surrogate_{ch}.pth")
        models.append(m)
        likelihoods.append(l)

    print(f"\nTotal GP training time: {time.time() - t0:.1f} s")

    np.savez("models/gp_cfg.npz",
             X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std,
             n_inducing=N_INDUCING)

    evaluate_gp(models, likelihoods, Xte, Yte_raw, Y_mean, Y_std)

    print("\n✓ GP Surrogate training complete")
    print("  Saved: models/gp_surrogate_0.pth ... models/gp_surrogate_5.pth")