
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

from quadrotor_simulator import delta_linear
from surrogate_common import (
    QuadrotorNNSurrogate, DATA_DIR, DEVICE,
    load_splits, input_norm_stats, residual_norm_stats, nonlinear_residual,
)

# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
BATCH_SIZE   = 1024
EPOCHS       = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────
#  Data loading and normalisation
# ─────────────────────────────────────────
def _normalise(X, Y, X_mean, X_std, Y_mean, Y_std):
    Xn = (X.astype(np.float32) - X_mean) / X_std
    Yn = (nonlinear_residual(X, Y).astype(np.float32) - Y_mean) / Y_std
    return Xn, Yn


def load_data(data_dir=DATA_DIR):
    """Load splits and compute physics-informed (residual) normalisation stats."""
    s = load_splits(data_dir)
    X_mean, X_std = input_norm_stats(s["X_train"])
    Y_mean, Y_std = residual_norm_stats(s["X_train"], s["Y_train"])

    Xtr, Ytr = _normalise(s["X_train"], s["Y_train"], X_mean, X_std, Y_mean, Y_std)
    Xvl, Yvl = _normalise(s["X_val"],   s["Y_val"],   X_mean, X_std, Y_mean, Y_std)
    Xte, Yte = _normalise(s["X_test"],  s["Y_test"],  X_mean, X_std, Y_mean, Y_std)
    Y_test_raw = nonlinear_residual(s["X_test"], s["Y_test"]).astype(np.float32)

    return (Xtr, Ytr), (Xvl, Yvl), (Xte, Yte, Y_test_raw), (X_mean, X_std, Y_mean, Y_std)


def make_loader(X, Y, shuffle=True, batch_size=BATCH_SIZE):
    ds = TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ─────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS,
                lr=LR, weight_decay=WEIGHT_DECAY, save_path="models/nn_surrogate.pth",
                verbose=True):
    """Train with AdamW + cosine-annealing LR. Saves the best-val checkpoint."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val = float("inf")
    train_losses, val_losses = [], []
    if verbose:
        print(f"\nTraining for {epochs} epochs...")
        print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
        print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total += loss.item() * len(xb)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                for xb, yb in val_loader
            ) / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"{epoch:6d}  {train_loss:12.6f}  {val_loss:12.6f}  "
                  f"{scheduler.get_last_lr()[0]:10.2e}")

    if verbose:
        print(f"\nBest validation loss: {best_val:.6f}")
    return train_losses, val_losses, best_val


def train_nn(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, seed=RANDOM_SEED,
             save_path=None, verbose=False):
    """
    Reusable trainer for the sample-efficiency sweep.
    Trains on (X, Y) physical data using residual normalisation derived from X,Y.
    Returns (NNSurrogate-ready model, cfg-dict).
    """
    torch.manual_seed(seed)
    X_mean, X_std = input_norm_stats(X)
    Y_mean, Y_std = residual_norm_stats(X, Y)
    Xn, Yn = _normalise(X, Y, X_mean, X_std, Y_mean, Y_std)

    n_val = max(1, int(0.1 * len(Xn)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(Xn))
    vi, ti = perm[:n_val], perm[n_val:]
    tr_loader = make_loader(Xn[ti], Yn[ti], shuffle=True, batch_size=batch_size)
    vl_loader = make_loader(Xn[vi], Yn[vi], shuffle=False, batch_size=batch_size)

    model = QuadrotorNNSurrogate().to(DEVICE)
    train_model(model, tr_loader, vl_loader, epochs=epochs,
                save_path=save_path, verbose=verbose)
    cfg = {"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}
    return model, cfg


# ─────────────────────────────────────────
#  Evaluation + plots
# ─────────────────────────────────────────
def evaluate_model(model, Xte, Y_test_raw, Y_mean, Y_std,
                   ckpt="models/nn_surrogate.pth"):
    """RMSE in physical residual units per channel."""
    model.eval()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    with torch.no_grad():
        pred_norm = model(torch.tensor(Xte).float().to(DEVICE)).cpu().numpy()
    pred_phys = pred_norm * Y_std + Y_mean
    true_phys = Y_test_raw

    names = ["d_phi", "d_theta", "d_psi", "d_p", "d_q", "d_r"]
    print("\n── Test-set nonlinear-residual RMSE (physical units) ──")
    for i, nm in enumerate(names):
        rmse = np.sqrt(np.mean((pred_phys[:, i] - true_phys[:, i]) ** 2))
        print(f"  {nm:9s}: RMSE = {rmse:.2e}")
    overall = np.sqrt(np.mean((pred_phys - true_phys) ** 2))
    print(f"\n  Overall nonlinear-residual RMSE : {overall:.2e}")
    return pred_phys, true_phys


def benchmark_inference(model, n_trials=10_000):
    from quadrotor_simulator import simulate_step
    x_dummy = torch.randn(1, 9).to(DEVICE)
    for _ in range(100):
        model(x_dummy)
    t = time.perf_counter()
    for _ in range(n_trials):
        with torch.no_grad():
            model(x_dummy)
    nn_us = (time.perf_counter() - t) / n_trials * 1e6

    state, u = np.zeros(6), np.zeros(3)
    t = time.perf_counter()
    for _ in range(n_trials):
        simulate_step(state, u)
    ode_us = (time.perf_counter() - t) / n_trials * 1e6

    print("\n── Inference Speed Benchmark ──")
    print(f"  NN surrogate  : {nn_us:.2f} us per call")
    print(f"  ODE (RK45)    : {ode_us:.2f} us per call")
    print(f"  Speedup       : {ode_us / nn_us:.1f}x")


def save_training_plot(train_losses, val_losses):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, label="Train Loss", linewidth=2)
    ax.semilogy(val_losses, label="Val Loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("NN Surrogate Training Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/nn_training_curve.png", dpi=150); plt.close()
    print("Saved: plots/nn_training_curve.png")


def save_parity_plot(pred_phys, true_phys):
    names = ["d_phi", "d_theta", "d_psi", "d_p", "d_q", "d_r"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7)); axes = axes.flatten()
    for i, (ax, nm) in enumerate(zip(axes, names)):
        ax.scatter(true_phys[:500, i], pred_phys[:500, i], alpha=0.3, s=8, color="steelblue")
        mn = min(true_phys[:, i].min(), pred_phys[:, i].min())
        mx = max(true_phys[:, i].max(), pred_phys[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
        r2 = np.corrcoef(true_phys[:, i], pred_phys[:, i])[0, 1] ** 2
        ax.set_title(f"{nm}  (R2 = {r2:.4f})")
        ax.set_xlabel("True"); ax.set_ylabel("Predicted"); ax.legend(fontsize=8)
    plt.suptitle("NN Surrogate — Predicted vs True Residuals (Test Set)", fontsize=13)
    plt.tight_layout(); plt.savefig("plots/nn_parity_plot.png", dpi=150); plt.close()
    print("Saved: plots/nn_parity_plot.png")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Training NN Surrogate for Quadrotor Attitude Dynamics")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)

    (Xtr, Ytr), (Xvl, Yvl), (Xte, Yte, Y_test_raw), (X_mean, X_std, Y_mean, Y_std) = load_data()
    print(f"Train: {len(Xtr):,}  |  Val: {len(Xvl):,}  |  Test: {len(Xte):,}")

    train_loader = make_loader(Xtr, Ytr, shuffle=True)
    val_loader   = make_loader(Xvl, Yvl, shuffle=False)

    model = QuadrotorNNSurrogate().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    train_losses, val_losses, _ = train_model(model, train_loader, val_loader)
    print(f"Training time: {time.time() - t0:.1f} s")

    np.savez("models/nn_surrogate_cfg.npz",
             X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

    pred_phys, true_phys = evaluate_model(model, Xte, Y_test_raw, Y_mean, Y_std)
    benchmark_inference(model)
    save_training_plot(train_losses, val_losses)
    save_parity_plot(pred_phys, true_phys)

    print("\n✓ NN Surrogate training complete")
    print("  Saved: models/nn_surrogate.pth, models/nn_surrogate_cfg.npz")
