"""
generate_dataset.py
===================
Generates 50,000 quadrotor attitude transition samples using
Latin Hypercube Sampling across the full flight envelope.

Saves three files to data/:
  - train.npz   (70% of data)
  - val.npz     (15% of data)
  - test.npz    (15% of data)

Each file contains:
  X : (N, 9)  — [state(6), input(3)]
  Y : (N, 6)  — residual delta_state = next_state - state
  Y_abs : (N, 6)  — absolute next state (for reference)
"""

import numpy as np
from scipy.stats import qmc
from joblib import Parallel, delayed
import os
import time
from tqdm import tqdm

from quadrotor_simulator import simulate_step, PARAMS


# ─────────────────────────────────────────
#  Sampling bounds (flight envelope)
# ─────────────────────────────────────────
BOUNDS = {
    # state bounds
    "phi":       (-0.524,  0.524),   # ±30 deg in rad
    "theta":     (-0.524,  0.524),   # ±30 deg
    "psi":       (-np.pi,  np.pi),   # full 360 yaw
    "p":         (-2.0,    2.0),     # roll rate rad/s
    "q":         (-2.0,    2.0),     # pitch rate rad/s
    "r":         (-1.0,    1.0),     # yaw rate rad/s
    # input bounds
    "tau_phi":   (-0.010,  0.010),   # roll torque N·m
    "tau_theta": (-0.010,  0.010),   # pitch torque N·m
    "tau_psi":   (-0.005,  0.005),   # yaw torque N·m
}

N_SAMPLES    = 50_000
N_JOBS       = 4          # parallel workers
RANDOM_SEED  = 42


def generate_samples(n_samples, seed=RANDOM_SEED):
    """
    Generate (state, input) pairs using Latin Hypercube Sampling.
    
    """
    print(f"Generating {n_samples:,} samples with Latin Hypercube Sampling...")

    # LHS in [0,1]^9
    sampler = qmc.LatinHypercube(d=9, seed=seed)
    unit_samples = sampler.random(n=n_samples)   # shape (N, 9)

    # Scale to physical bounds
    lower = np.array([v[0] for v in BOUNDS.values()])
    upper = np.array([v[1] for v in BOUNDS.values()])
    samples = qmc.scale(unit_samples, lower, upper)

    return samples   # columns: [phi, theta, psi, p, q, r, tau_phi, tau_theta, tau_psi]


def simulate_one(row):
    """Simulate a single state-input pair. Used for parallel processing."""
    state = row[:6]
    u     = row[6:]
    next_state = simulate_step(state, u)
    delta      = next_state - state
    return next_state, delta


def build_dataset(n_samples=N_SAMPLES):
    """Full pipeline: sample → simulate → return X, Y, Y_abs."""
    t0 = time.time()

    # 1. Generate samples
    X = generate_samples(n_samples)

    # 2. Simulate in parallel (much faster than sequential)
    print(f"Simulating {n_samples:,} transitions using {N_JOBS} parallel workers...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(simulate_one)(X[i]) for i in tqdm(range(n_samples), desc="Simulating")
    )

    Y_abs = np.array([r[0] for r in results])   # next states
    Y     = np.array([r[1] for r in results])   # deltas (residuals)

    t1 = time.time()
    print(f"\nDataset generated in {t1 - t0:.1f} seconds")
    print(f"X shape : {X.shape}   (samples × [state, input])")
    print(f"Y shape : {Y.shape}   (samples × delta_state)")

    return X, Y, Y_abs


def split_and_save(X, Y, Y_abs, save_dir="data"):
    """Split into train/val/test and save."""
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    N = len(X)
    idx = rng.permutation(N)

    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    # remaining goes to test

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    splits = {
        "train": train_idx,
        "val":   val_idx,
        "test":  test_idx,
    }

    for split_name, split_idx in splits.items():
        np.savez(
            os.path.join(save_dir, f"{split_name}.npz"),
            X=X[split_idx],
            Y=Y[split_idx],
            Y_abs=Y_abs[split_idx],
        )
        print(f"Saved {split_name:5s}: {len(split_idx):6,} samples → data/{split_name}.npz")

    # Also save normalisation statistics (computed from train set only)
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    np.savez(
        os.path.join(save_dir, "normstats.npz"),
        X_mean=X_train.mean(axis=0),
        X_std=X_train.std(axis=0)  + 1e-8,
        Y_mean=Y_train.mean(axis=0),
        Y_std=Y_train.std(axis=0)  + 1e-8,
    )
    print("Saved normstats.npz (mean/std for normalisation)")


def print_dataset_stats(X, Y):
    """Print a quick statistical summary of the dataset."""
    names = ["phi", "theta", "psi", "p", "q", "r",
             "tau_phi", "tau_theta", "tau_psi"]
    print("\n── Input Statistics ──")
    for i, name in enumerate(names):
        print(f"  {name:12s}: min={X[:,i].min():.4f}  max={X[:,i].max():.4f}  "
              f"mean={X[:,i].mean():.4f}  std={X[:,i].std():.4f}")

    delta_names = ["Δphi", "Δtheta", "Δpsi", "Δp", "Δq", "Δr"]
    print("\n── Residual (delta) Statistics ──")
    for i, name in enumerate(delta_names):
        print(f"  {name:8s}: min={Y[:,i].min():.6f}  max={Y[:,i].max():.6f}  "
              f"std={Y[:,i].std():.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Generation — Quadrotor Attitude Transitions")
    print("=" * 60)

    X, Y, Y_abs = build_dataset(N_SAMPLES)
    print_dataset_stats(X, Y)
    split_and_save(X, Y, Y_abs)

    print("\n✓ Dataset generation complete")
    print("  Files saved: data/train.npz, data/val.npz, data/test.npz, data/normstats.npz")