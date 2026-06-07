"""
train_nn_surrogate.py
=====================
Trains a Residual Neural Network surrogate for quadrotor attitude dynamics.

Architecture : Linear(9→128) → SiLU → Linear(128→128) → SiLU →
               Linear(128→64) → SiLU → Linear(64→6)

Input  : normalised [phi, theta, psi, p, q, r, tau_phi, tau_theta, tau_psi]
Output : delta_state = next_state - state  (in normalised space)

Saves:
  models/nn_surrogate.pth     — best model weights
  models/nn_surrogate_cfg.npz — normalisation stats embedded with model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import os
import time

# Placeholder for delta_linear. Replace this with your actual physics-based linear model function.
def delta_linear(X):
    """
    Computes the linear base model prediction for delta_state.
    Expects raw/unnormalised input X of shape (N, 9).
    Returns an array of shape (N, 6).
    """
    # TODO: Ensure your actual linear base mapping is implemented here
    return np.zeros((X.shape[0], 6), dtype=np.float32)


# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
BATCH_SIZE   = 1024
EPOCHS       = 150
LR           = 3e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIMS  = [128, 128, 64] # Aligned with docstring architecture
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────
#  Model definition
# ─────────────────────────────────────────
class QuadrotorNNSurrogate(nn.Module):
    """
    Feedforward neural network surrogate for quadrotor dynamics.
    Predicts normalized nonlinear delta_state residual given (state, input).
    """

    def __init__(self, in_dim=9, out_dim=6, hidden_dims=HIDDEN_DIMS):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.SiLU()]   # Aligned with docstring SiLU activation
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

        # Weight initialisation: He uniform (good for SiLU)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
#  Data loading and normalisation
# ─────────────────────────────────────────
def load_data():
    """Load dataset splits and calculate physics-informed normalisation statistics."""
    train = np.load(r"D:\MINOR\quadrotor_surrogate_mpc\data\train.npz")
    val   = np.load(r"D:\MINOR\quadrotor_surrogate_mpc\data\val.npz")
    test  = np.load(r"D:\MINOR\quadrotor_surrogate_mpc\data\test.npz")
    stats = np.load(r"D:\MINOR\quadrotor_surrogate_mpc\data\normstats.npz")
    
    X_mean = stats["X_mean"].astype(np.float32)
    X_std  = stats["X_std"].astype(np.float32)

    # Physics-informed: target = full residual - linear base
    Y_nl_train = train["Y"] - delta_linear(train["X"])
    Y_mean = Y_nl_train.mean(axis=0).astype(np.float32)
    Y_std  = (Y_nl_train.std(axis=0) + 1e-8).astype(np.float32)

    def normalise(X, Y):
        Xn = (X.astype(np.float32) - X_mean) / X_std
        Yn = ((Y.astype(np.float32) - delta_linear(X)) - Y_mean) / Y_std
        return Xn, Yn

    Xtr, Ytr = normalise(train["X"], train["Y"])
    Xvl, Yvl = normalise(val["X"],   val["Y"])
    Xte, Yte = normalise(test["X"],  test["Y"])

    # Test set: extract raw nonlinear residual for residual-space physical evaluation
    Y_test_raw = (test["Y"] - delta_linear(test["X"])).astype(np.float32)

    return (Xtr, Ytr), (Xvl, Yvl), (Xte, Yte, Y_test_raw), (X_mean, X_std, Y_mean, Y_std)


def make_loader(X, Y, shuffle=True):
    ds = TensorDataset(torch.tensor(X), torch.tensor(Y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


# ─────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────
def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    """Train with AdamW + cosine annealing LR schedule."""
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item() * len(xb)
        train_loss = total_loss / len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                criterion(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
                for xb, yb in val_loader
            ) / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "models/nn_surrogate.pth")

        if epoch % 20 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"{epoch:6d}  {train_loss:12.6f}  {val_loss:12.6f}  {lr_now:10.2e}")

    print(f"\nBest validation loss: {best_val_loss:.6f}")
    return train_losses, val_losses


# ─────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────
def evaluate_model(model, Xte, Yte_norm, Y_test_raw, Y_mean, Y_std):
    """Compute RMSE in physical units for each state channel (Nonlinear residual space)."""
    model.eval()
    model.load_state_dict(torch.load("models/nn_surrogate.pth", map_location=DEVICE))

    with torch.no_grad():
        pred_norm = model(torch.tensor(Xte).to(DEVICE)).cpu().numpy()

    # Denormalise back to physical unmodeled/nonlinear units
    pred_phys = pred_norm * Y_std + Y_mean
    true_phys = Y_test_raw

    channel_names = ["Δphi_nl", "Δtheta_nl", "Δpsi_nl", "Δp_nl", "Δq_nl", "Δr_nl"]
    print("\n── Test Set Nonlinear Residual RMSE (physical units) ──")
    for i, name in enumerate(channel_names):
        rmse = np.sqrt(np.mean((pred_phys[:, i] - true_phys[:, i])**2))
        print(f"  {name:11s}: RMSE = {rmse:.2e}")

    overall_rmse = np.sqrt(np.mean((pred_phys - true_phys)**2))
    print(f"\n  Overall Nonlinear Residual RMSE : {overall_rmse:.2e}")
    return pred_phys, true_phys


# ─────────────────────────────────────────
#  Inference speed benchmark
# ─────────────────────────────────────────
def benchmark_inference(model, n_trials=10_000):
    """Measure how fast a single forward pass is (vs ODE integration)."""
    import time
    from quadrotor_simulator import simulate_step

    x_dummy = torch.randn(1, 9).to(DEVICE)

    # Warmup
    for _ in range(100):
        model(x_dummy)

    # NN inference
    start = time.perf_counter()
    for _ in range(n_trials):
        with torch.no_grad():
            model(x_dummy)
    nn_time = (time.perf_counter() - start) / n_trials * 1e6  # microseconds

    # ODE integration
    state = np.zeros(6)
    u     = np.zeros(3)
    start = time.perf_counter()
    for _ in range(n_trials):
        simulate_step(state, u)
    ode_time = (time.perf_counter() - start) / n_trials * 1e6

    print(f"\n── Inference Speed Benchmark ──")
    print(f"  NN surrogate  : {nn_time:.2f} μs per call")
    print(f"  ODE (RK45)    : {ode_time:.2f} μs per call")
    print(f"  Speedup       : {ode_time / nn_time:.1f}×")


# ─────────────────────────────────────────
#  Save plots
# ─────────────────────────────────────────
def save_training_plot(train_losses, val_losses):
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(train_losses, label="Train Loss", linewidth=2)
    ax.semilogy(val_losses,   label="Val Loss",   linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("NN Surrogate Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/nn_training_curve.png", dpi=150)
    plt.close()
    print("Saved: plots/nn_training_curve.png")


def save_parity_plot(pred_phys, true_phys):
    """Predicted vs True scatter plot for each nonlinear residual channel."""
    channel_names = ["Δφ_nl", "Δθ_nl", "Δψ_nl", "Δp_nl", "Δq_nl", "Δr_nl"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        ax.scatter(true_phys[:500, i], pred_phys[:500, i],
                   alpha=0.3, s=8, color="steelblue")
        mn = min(true_phys[:, i].min(), pred_phys[:, i].min())
        mx = max(true_phys[:, i].max(), pred_phys[:, i].max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect")
        ax.set_title(f"{name}  (R² = {np.corrcoef(true_phys[:,i], pred_phys[:,i])[0,1]**2:.4f})")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.legend(fontsize=8)
    plt.suptitle("NN Surrogate — Predicted vs True Residuals (Test Set)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/nn_parity_plot.png", dpi=150)
    plt.close()
    print("Saved: plots/nn_parity_plot.png")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Training NN Surrogate for Quadrotor Attitude Dynamics")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Load data
    (Xtr, Ytr), (Xvl, Yvl), (Xte, Yte, Y_test_raw), (X_mean, X_std, Y_mean, Y_std) = load_data()
    print(f"Train: {len(Xtr):,}  |  Val: {len(Xvl):,}  |  Test: {len(Xte):,}")

    train_loader = make_loader(Xtr, Ytr, shuffle=True)
    val_loader   = make_loader(Xvl, Yvl, shuffle=False)

    # Build and train model
    model = QuadrotorNNSurrogate().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    t0 = time.time()
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    print(f"Training time: {time.time() - t0:.1f} s")

    # Save normalisation stats with model
    np.savez("models/nn_surrogate_cfg.npz",
             X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

    # Evaluate
    pred_phys, true_phys = evaluate_model(model, Xte, Yte, Y_test_raw, Y_mean, Y_std)

    # Benchmark speed
    benchmark_inference(model)


    # Plots
    save_training_plot(train_losses, val_losses)
    save_parity_plot(pred_phys, true_phys)

    print("\n✓ NN Surrogate training complete")
    print("  Saved: models/nn_surrogate.pth, models/nn_surrogate_cfg.npz")