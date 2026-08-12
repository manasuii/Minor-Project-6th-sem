"""
evaluate_surrogates.py
======================
Open-loop three-way surrogate evaluation (Objective O2 + headline contribution):
ResNN vs Gaussian Process vs Polynomial Chaos Expansion.

Tier-1 verification metrics from the proposal:
  1. One-step accuracy   — RMSE per state channel + overall (full one-step delta)
  2. Rollout stability    — error growth over an unguided 5 s horizon vs true ODE
  3. Sample efficiency     — test RMSE vs training-set size {1k, 5k, 10k, 25k}
  4. Computational latency — mean wall-clock per prediction

Outputs:
  plots/rollout_error.png
  plots/sample_efficiency.png
  plots/latency_comparison.png
  results/surrogate_comparison.csv
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from quadrotor_simulator import simulate_trajectory, PARAMS
from surrogate_common import load_splits, load_subset, load_all_available, CHANNEL_NAMES

# reusable trainers for the sample-efficiency sweep
from train_nn_surrogate import train_nn
from train_gp_surrogate import train_gp
from train_pce_surrogate import fit_pce, LOWER, UPPER
from surrogate_common import NNSurrogate, GPSurrogate, PCESurrogate

DT = PARAMS["dt"]

# ── sweep configuration (reduced epochs keep the sweep within budget) ──
RUN_SAMPLE_EFFICIENCY = True
SAMPLE_SIZES   = [1000, 5000, 10000, 25000]
SWEEP_NN_EPOCHS = 60
SWEEP_GP_EPOCHS = 40
SWEEP_GP_INDUCING = 256
MODEL_COLORS = {"ResNN": "#2563EB", "GP": "#10B981", "PCE": "#F59E0B"}


# ════════════════════════════════════════════════════════════════════════
#  Metric 1 — one-step RMSE (full delta) on the test set
# ════════════════════════════════════════════════════════════════════════
def one_step_rmse_table(surrogates, X_test, Y_test_full):
    print("\n── One-step prediction RMSE (full delta, physical units) ──")
    header = f"{'Channel':9s}" + "".join(f"{name:>13s}" for name in surrogates)
    print(header); print("-" * len(header))
    per_model_pred = {name: s.predict_delta(X_test) for name, s in surrogates.items()}

    overall = {}
    for ch, cname in enumerate(CHANNEL_NAMES):
        row = f"{cname:9s}"
        for name in surrogates:
            rmse = np.sqrt(np.mean((per_model_pred[name][:, ch] - Y_test_full[:, ch]) ** 2))
            row += f"{rmse:13.2e}"
        print(row)
    print("-" * len(header))
    row = f"{'OVERALL':9s}"
    for name in surrogates:
        o = np.sqrt(np.mean((per_model_pred[name] - Y_test_full) ** 2))
        overall[name] = o
        row += f"{o:13.2e}"
    print(row)
    return overall


# ════════════════════════════════════════════════════════════════════════
#  Metric 2 — multi-step rollout vs true ODE
# ════════════════════════════════════════════════════════════════════════
def _batched_rollout(surrogate, X0, U_seqs):
    """Step all trajectories together. X0:(n,6), U_seqs:(n,T,3) -> (n,T+1,6)."""
    n, T = X0.shape[0], U_seqs.shape[1]
    states = np.zeros((n, T + 1, 6))
    states[:, 0] = X0
    for t in range(T):
        xu = np.concatenate([states[:, t], U_seqs[:, t]], axis=1)   # (n, 9)
        states[:, t + 1] = states[:, t] + surrogate.predict_delta(xu)
    return states


def run_rollout_comparison(surrogates, n_traj=20, T_rollout=100, seed=42):
    """Mean absolute rollout error per channel vs the true ODE, for each model."""
    rng = np.random.default_rng(seed)
    X0 = rng.uniform([-0.3, -0.3, -1.0, -1.0, -1.0, -0.5],
                     [0.3, 0.3, 1.0, 1.0, 1.0, 0.5], size=(n_traj, 6))
    u_base = rng.uniform(-0.005, 0.005, (n_traj, 3))
    U_seqs = (u_base[:, None, :] + rng.normal(0, 0.001, (n_traj, T_rollout, 3)))
    U_seqs = np.clip(U_seqs, -0.01, 0.01)

    true = np.stack([simulate_trajectory(X0[i], U_seqs[i]) for i in range(n_traj)])

    per_model_mean, per_model_std = {}, {}
    for name, s in surrogates.items():
        pred = _batched_rollout(s, X0, U_seqs)
        err = np.abs(pred - true)                 # (n, T+1, 6)
        per_model_mean[name] = err.mean(axis=0)   # (T+1, 6)
        per_model_std[name]  = err.std(axis=0)
    return per_model_mean, per_model_std, T_rollout


def plot_rollout_error(per_model_mean, per_model_std, T_rollout):
    t = np.arange(T_rollout + 1) * DT
    chan = ["phi (roll)", "theta (pitch)", "psi (yaw)", "p", "q", "r"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    for i, (ax, nm) in enumerate(zip(axes, chan)):
        for name, mean in per_model_mean.items():
            col = MODEL_COLORS.get(name, None)
            ax.plot(t, mean[:, i], color=col, linewidth=2, label=name)
            ax.fill_between(t, mean[:, i] - per_model_std[name][:, i],
                            mean[:, i] + per_model_std[name][:, i],
                            alpha=0.12, color=col)
        ax.set_title(nm); ax.set_xlabel("Time (s)")
        ax.set_ylabel("Abs error (rad or rad/s)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Surrogate rollout error over 5 s (mean ±1 std over 20 trajectories)",
                 fontsize=13)
    plt.tight_layout(); plt.savefig("plots/rollout_error.png", dpi=150); plt.close()
    print("Saved: plots/rollout_error.png")


def rollout_summary(per_model_mean, T_rollout):
    idx = {0.5: min(int(0.5 / DT), T_rollout),
           2.0: min(int(2.0 / DT), T_rollout),
           5.0: T_rollout}
    print("\n── Rollout error (overall mean abs error across channels) ──")
    print(f"{'Model':9s}  {'@0.5s':>10s}  {'@2.0s':>10s}  {'@5.0s':>10s}")
    out = {}
    for name, mean in per_model_mean.items():
        overall = mean.mean(axis=1)
        out[name] = float(overall[idx[5.0]])
        print(f"{name:9s}  {overall[idx[0.5]]:10.2e}  {overall[idx[2.0]]:10.2e}  "
              f"{overall[idx[5.0]]:10.2e}")
    return out


# ════════════════════════════════════════════════════════════════════════
#  Metric 3 — sample-efficiency curve
# ════════════════════════════════════════════════════════════════════════
def _rmse_full_delta(surrogate, X_test, Y_test_full):
    return float(np.sqrt(np.mean((surrogate.predict_delta(X_test) - Y_test_full) ** 2)))


def run_sample_efficiency(X_test, Y_test_full, sizes=SAMPLE_SIZES):
    print("\n── Sample-efficiency sweep (retraining each model on subsets) ──")
    curves = {"ResNN": [], "GP": [], "PCE": []}
    for n in sizes:
        Xs, Ys = load_subset(n)
        print(f"\n  n_train = {n:,}")

        # ResNN
        t0 = time.time()
        nn_model, nn_cfg = train_nn(Xs, Ys, epochs=SWEEP_NN_EPOCHS, save_path=None)
        nn_surr = NNSurrogate(model=nn_model, cfg=nn_cfg)
        r = _rmse_full_delta(nn_surr, X_test, Y_test_full)
        curves["ResNN"].append(r)
        print(f"    ResNN : RMSE={r:.2e}  ({time.time()-t0:.0f}s)")

        # GP
        t0 = time.time()
        gp_models, gp_liks, gp_cfg = train_gp(
            Xs, Ys, epochs=SWEEP_GP_EPOCHS, n_inducing=SWEEP_GP_INDUCING)
        gp_surr = GPSurrogate(models=gp_models, likelihoods=gp_liks, cfg=gp_cfg)
        r = _rmse_full_delta(gp_surr, X_test, Y_test_full)
        curves["GP"].append(r)
        print(f"    GP    : RMSE={r:.2e}  ({time.time()-t0:.0f}s)")

        # PCE
        t0 = time.time()
        approx, _ = fit_pce(Xs, Ys)
        pce_surr = PCESurrogate(approx=approx, lower=LOWER, upper=UPPER)
        r = _rmse_full_delta(pce_surr, X_test, Y_test_full)
        curves["PCE"].append(r)
        print(f"    PCE   : RMSE={r:.2e}  ({time.time()-t0:.0f}s)")
    return sizes, curves


def plot_sample_efficiency(sizes, curves):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ys in curves.items():
        if ys:
            ax.loglog(sizes, ys, "o-", color=MODEL_COLORS.get(name),
                      linewidth=2, markersize=6, label=name)
    ax.set_xlabel("Training-set size"); ax.set_ylabel("Test RMSE (full delta)")
    ax.set_title("Sample Efficiency — RMSE vs Training Set Size")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/sample_efficiency.png", dpi=150); plt.close()
    print("Saved: plots/sample_efficiency.png")


# ════════════════════════════════════════════════════════════════════════
#  Metric 4 — inference latency
# ════════════════════════════════════════════════════════════════════════
def benchmark_latency(surrogates, n_trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    xu = np.concatenate([rng.uniform(-0.3, 0.3, (1, 6)),
                         rng.uniform(-0.01, 0.01, (1, 3))], axis=1)
    print("\n── Inference latency (single-step predict_delta) ──")
    out = {}
    for name, s in surrogates.items():
        for _ in range(50):                  # warmup
            s.predict_delta(xu)
        t0 = time.perf_counter()
        for _ in range(n_trials):
            s.predict_delta(xu)
        us = (time.perf_counter() - t0) / n_trials * 1e6
        out[name] = us
        print(f"  {name:9s}: {us:10.2f} us/call")
    return out


def plot_latency(latency):
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(latency.keys())
    ax.bar(names, [latency[n] for n in names],
           color=[MODEL_COLORS.get(n, "#888") for n in names])
    ax.set_ylabel("Latency (us/call)")
    ax.set_title("Surrogate Inference Latency")
    ax.set_yscale("log"); ax.grid(True, axis="y", alpha=0.3)
    for i, n in enumerate(names):
        ax.text(i, latency[n], f"{latency[n]:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig("plots/latency_comparison.png", dpi=150); plt.close()
    print("Saved: plots/latency_comparison.png")


# ════════════════════════════════════════════════════════════════════════
#  Summary table
# ════════════════════════════════════════════════════════════════════════
def save_summary(onestep, rollout5s, latency, sample_curves, sizes):
    import pandas as pd
    rows = []
    for name in onestep:
        row = {
            "Surrogate": name,
            "1-step RMSE": f"{onestep[name]:.3e}",
            "Rollout err @5s": f"{rollout5s.get(name, float('nan')):.3e}",
            "Latency (us)": round(latency.get(name, float("nan")), 2),
        }
        if sample_curves and sample_curves.get(name):
            for n, r in zip(sizes, sample_curves[name]):
                row[f"RMSE @{n}"] = f"{r:.3e}"
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/surrogate_comparison.csv", index=False)
    print("\n── Three-way surrogate comparison ──")
    print(df.to_string(index=False))
    print("\nSaved: results/surrogate_comparison.csv")


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Surrogate Evaluation — ResNN vs GP vs PCE (open-loop)")
    print("=" * 60)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("\nLoading trained surrogates...")
    surrogates = load_all_available()
    if not surrogates:
        raise SystemExit("No trained surrogates found. Train NN/GP/PCE first.")

    s = load_splits()
    X_test, Y_test_full = s["X_test"], s["Y_test"]

    onestep = one_step_rmse_table(surrogates, X_test, Y_test_full)

    print("\nRunning rollout comparison (20 trajectories × 100 steps)...")
    mean_err, std_err, T = run_rollout_comparison(surrogates)
    plot_rollout_error(mean_err, std_err, T)
    rollout5s = rollout_summary(mean_err, T)

    latency = benchmark_latency(surrogates)
    plot_latency(latency)

    sample_sizes, sample_curves = [], {}
    if RUN_SAMPLE_EFFICIENCY:
        sample_sizes, sample_curves = run_sample_efficiency(X_test, Y_test_full)
        plot_sample_efficiency(sample_sizes, sample_curves)

    save_summary(onestep, rollout5s, latency, sample_curves, sample_sizes)

    print("\n✓ Surrogate evaluation complete")
