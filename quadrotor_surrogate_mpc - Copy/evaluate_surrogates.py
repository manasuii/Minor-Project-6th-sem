"""
evaluate_surrogates.py
======================
Comprehensive surrogate evaluation:
  1. Multi-step rollout comparison (surrogate vs true ODE)
  2. Sample efficiency curve (RMSE vs training set size)
  3. Summary comparison table

Generates plots saved to plots/
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from quadrotor_simulator import simulate_trajectory, PARAMS
from train_nn_surrogate import QuadrotorNNSurrogate
from mpc_surrogate import delta_linear


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = PARAMS["dt"]


# ─────────────────────────────────────────
#  Load the trained NN surrogate
# ─────────────────────────────────────────
def load_nn_surrogate():
    """Load trained NN surrogate and normalisation stats."""
    cfg = np.load("models/nn_surrogate_cfg.npz")
    X_mean = cfg["X_mean"].astype(np.float32)
    X_std  = cfg["X_std"].astype(np.float32)
    Y_mean = cfg["Y_mean"].astype(np.float32)
    Y_std  = cfg["Y_std"].astype(np.float32)

    model = QuadrotorNNSurrogate().to(DEVICE)
    model.load_state_dict(torch.load("models/nn_surrogate.pth", map_location=DEVICE))
    model.eval()

    return model, X_mean, X_std, Y_mean, Y_std


def nn_predict_step(model, state, u, X_mean, X_std, Y_mean, Y_std):
    """One-step prediction using the NN surrogate with embedded physics."""
    xu = np.concatenate([state, u]).astype(np.float32)
    xu_norm = (xu - X_mean) / X_std
    with torch.no_grad():
        delta_norm = model(torch.tensor(xu_norm).unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
    
    # Physics-informed reconstruction: NN correction + linear base model step
    delta = delta_norm * Y_std + Y_mean + delta_linear(np.concatenate([state, u]).reshape(1, -1))[0]
    return state + delta


# ─────────────────────────────────────────
#  Rollout comparison
# ─────────────────────────────────────────
def rollout_with_nn(x0, u_sequence, model, X_mean, X_std, Y_mean, Y_std):
    """Roll out a trajectory using the NN surrogate."""
    T = len(u_sequence)
    states = np.zeros((T + 1, 6))
    states[0] = x0
    for t in range(T):
        states[t + 1] = nn_predict_step(
            model, states[t], u_sequence[t], X_mean, X_std, Y_mean, Y_std
        )
    return states


def run_rollout_comparison(n_trajectories=20, T_rollout=100):
    """
    Compare NN surrogate vs true ODE on multiple random trajectories.
    T_rollout = 100 steps = 5 seconds
    """
    model, X_mean, X_std, Y_mean, Y_std = load_nn_surrogate()

    rng = np.random.default_rng(42)
    rollout_errors = []

    for traj_idx in range(n_trajectories):
        # Random initial state within the flight envelope
        x0 = rng.uniform(
            [-0.3, -0.3, -1.0, -1.0, -1.0, -0.5],
            [ 0.3,  0.3,  1.0,  1.0,  1.0,  0.5]
        )
        # Random but smooth control sequence
        u_base = rng.uniform(-0.005, 0.005, 3)
        u_seq  = np.tile(u_base, (T_rollout, 1)) + \
                 rng.normal(0, 0.001, (T_rollout, 3))
        u_seq  = np.clip(u_seq, -0.01, 0.01)

        # True trajectory
        true_traj = simulate_trajectory(x0, u_seq)

        # NN surrogate trajectory
        nn_traj = rollout_with_nn(x0, u_seq, model, X_mean, X_std, Y_mean, Y_std)

        # Per-step absolute error
        error = np.abs(nn_traj - true_traj)   # (T+1, 6)
        rollout_errors.append(error)

    rollout_errors = np.array(rollout_errors)   # (n_traj, T+1, 6)
    mean_error  = rollout_errors.mean(axis=0)   # (T+1, 6) — mean over trajectories
    std_error   = rollout_errors.std(axis=0)

    return mean_error, std_error, T_rollout


def plot_rollout_error(mean_error, std_error, T_rollout):
    """Plot mean absolute rollout error over time for each channel."""
    time_axis  = np.arange(T_rollout + 1) * DT
    chan_names = ["φ (roll)", "θ (pitch)", "ψ (yaw)", "p", "q", "r"]
    colors     = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    for i, (ax, name, col) in enumerate(zip(axes, chan_names, colors)):
        ax.plot(time_axis, mean_error[:, i], color=col, linewidth=2, label="Mean error")
        ax.fill_between(
            time_axis,
            mean_error[:, i] - std_error[:, i],
            mean_error[:, i] + std_error[:, i],
            alpha=0.2, color=col, label="±1 std"
        )
        ax.set_title(name)
        ax.set_ylabel("Abs Error (rad or rad/s)")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("NN Surrogate — Rollout Error over 5 Seconds (20 trajectories)", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/rollout_error.png", dpi=150)
    plt.close()
    print("Saved: plots/rollout_error.png")


# ─────────────────────────────────────────
#  Print summary
# ─────────────────────────────────────────
def print_evaluation_summary(mean_error, T_rollout):
    chan_names = ["φ", "θ", "ψ", "p", "q", "r"]
    print("\n── Rollout Error Summary (mean abs error) ──")
    print(f"{'Channel':8s}  {'@0.5s':10s}  {'@2.0s':10s}  {'@5.0s':10s}")
    print("-" * 45)
    for i, name in enumerate(chan_names):
        t05 = int(0.5 / DT)
        t20 = int(2.0 / DT)
        t50 = int(5.0 / DT)
        print(f"{name:8s}  {mean_error[t05, i]:.2e}    {mean_error[t20, i]:.2e}    {mean_error[t50, i]:.2e}")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Surrogate Rollout Validation")
    print("=" * 60)

    os.makedirs("plots", exist_ok=True)

    print("\nRunning rollout comparison (20 trajectories × 100 steps)...")
    mean_error, std_error, T_rollout = run_rollout_comparison(
        n_trajectories=20, T_rollout=100
    )

    print_evaluation_summary(mean_error, T_rollout)
    plot_rollout_error(mean_error, std_error, T_rollout)

    print("\n✓ Rollout evaluation complete")
    print("  Plots saved to plots/rollout_error.png")