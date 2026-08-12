import subprocess
import sys
import time

steps = [
    ("validate_simulator.py",   "Step 1/9: Validating Simulator"),
    ("generate_dataset.py",     "Step 2/9: Generating Dataset"),
    ("train_nn_surrogate.py",   "Step 3/9: Training NN Surrogate"),
    ("train_gp_surrogate.py",   "Step 4/9: Training GP Surrogate"),
    ("train_pce_surrogate.py",  "Step 5/9: Training PCE Surrogate"),
    ("evaluate_surrogates.py",  "Step 6/9: Evaluating Surrogates (3-way)"),
    ("mpc_surrogate.py",        "Step 7/9: Building Surrogate-MPC"),
    ("baseline_controllers.py", "Step 8/9: Testing Baselines"),
    ("run_experiments.py",      "Step 9/9: Running All Experiments"),
]

total_start = time.time()
for script, label in steps:
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    t0 = time.time()
    subprocess.run([sys.executable, script], check=True)
    print(f"\n  Completed in {(time.time() - t0)/60:.1f} minutes")

print(f"\n{'='*60}")
print(f"  FULL PIPELINE COMPLETE in {(time.time() - total_start)/60:.1f} minutes")
print(f"{'='*60}")
print("\nCheck these folders:")
print("  plots/    — all figures (training, rollout, sample-efficiency, latency, exp1-6)")
print("  results/  — surrogate_comparison.csv, summary_table.csv, realtime_feasibility.csv")
print("  models/   — nn_surrogate.pth, gp_surrogate_*.pth, pce_surrogate.pkl")
print("  data/     — train/val/test .npz")
