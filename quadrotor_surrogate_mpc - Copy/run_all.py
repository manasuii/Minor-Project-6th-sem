"""
run_all.py
==========
Master script: runs the complete project pipeline end-to-end.
Run this after installing dependencies (Step 1).

Execution order:
  1. generate_dataset.py   — create 50k transition samples
  2. train_nn_surrogate.py — train neural network surrogate
  3. train_gp_surrogate.py — train Gaussian Process surrogate
  4. evaluate_surrogates.py — rollout validation
  5. mpc_surrogate.py      — build and test MPC
  6. baseline_controllers.py — test baselines
  7. run_experiments.py    — all closed-loop experiments
"""

import subprocess
import sys
import time

steps = [
    ("generate_dataset.py",     "Step 1/7: Generating Dataset"),
    ("train_nn_surrogate.py",   "Step 2/7: Training NN Surrogate"),
    ("train_gp_surrogate.py",   "Step 3/7: Training GP Surrogate"),
    ("evaluate_surrogates.py",  "Step 4/7: Evaluating Surrogates"),
    ("mpc_surrogate.py",        "Step 5/7: Building MPC"),
    ("baseline_controllers.py", "Step 6/7: Testing Baselines"),
    ("run_experiments.py",      "Step 7/7: Running All Experiments"),
]

total_start = time.time()

for script, label in steps:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} minutes")

total_elapsed = (time.time() - total_start) / 60
print(f"\n{'='*60}")
print(f"  FULL PIPELINE COMPLETE in {total_elapsed:.1f} minutes")
print(f"{'='*60}")
print("\nCheck these folders:")
print("  plots/   — all figures")
print("  results/ — summary_table.csv")
print("  models/  — saved surrogate models")
print("  data/    — training dataset")