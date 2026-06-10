"""
run_experiments.py
==================
Runs all closed-loop experiments and generates paper-ready results.

Experiments:
  1. Step response      (roll 0 → 20°)
  2. Sine tracking      (15° sin reference)
  3. Disturbance reject (impulse torque at t=3s)
  4. Multi-axis         (simultaneous roll + pitch step)
  5. Model mismatch     (Ixx varies ±30%)

Saves:
  plots/exp1_step_response.png
  plots/exp2_sine_tracking.png
  plots/exp3_disturbance.png
  plots/exp4_multiaxis.png
  plots/exp5_mismatch.png
  results/summary_table.csv
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from quadrotor_simulator import simulate_step, PARAMS
from mpc_surrogate import SurrogateMPC
from baseline_controllers import PIDController, LinearisedMPC


os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)

DT = PARAMS["dt"]


# ─────────────────────────────────────────
#  Closed-loop simulation runner
# ─────────────────────────────────────────
def run_closed_loop(controller, x0, ref_fn, T_steps, disturb_fn=None, plant_params=None):
    """
    Generic closed-loop simulation.

    Parameters
    ----------
    controller : object with .compute(state, ref) or .solve(state, ref)
    x0         : array (6,)      — initial state
    ref_fn     : callable(t)     — reference as function of timestep index
    T_steps    : int             — number of steps to simulate
    disturb_fn : callable(t, u)  — optional disturbance applied to input
    plant_params : dict          — optional custom plant parameters (mismatch test)

    Returns
    -------
    states     : array (T+1, 6)
    inputs     : array (T, 3)
    solve_times: list of float (ms)
    """
    params = plant_params if plant_params is not None else PARAMS
    states      = np.zeros((T_steps + 1, 6))
    inputs      = np.zeros((T_steps, 3))
    solve_times = []
    states[0]   = x0

    # Reset PID integral if applicable
    if hasattr(controller, "reset"):
        controller.reset()

    for t in range(T_steps):
        ref   = ref_fn(t)
        state = states[t]

        t0 = time.perf_counter()
        if hasattr(controller, "solve"):      # MPC interface
            u, ms, _ = controller.solve(state, ref)
        else:                                 # PID interface
            u  = controller.compute(state, ref)
            ms = (time.perf_counter() - t0) * 1000

        if disturb_fn is not None:
            u = disturb_fn(t, u)

        u = np.clip(u, [-0.010, -0.010, -0.005], [0.010, 0.010, 0.005])

        inputs[t]      = u
        solve_times.append(ms)
        states[t + 1]  = simulate_step(state, u, params=params)

    return states, inputs, solve_times


# ─────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────
def compute_metrics(states, ref_states, inputs, channel=0):
    """
    Compute standard control performance metrics.

    Returns dict with: settling_time, overshoot_pct, tracking_rmse, ctrl_effort
    """
    errors = states[:, channel] - ref_states[:, channel]
    steady = ref_states[-1, channel]

    # Settling time: first time error stays within 2% of reference
    tol = 0.02 * abs(steady) if abs(steady) > 1e-6 else 0.005
    settled_idx = None
    for i in range(len(errors) - 1, -1, -1):
        if abs(errors[i]) > tol:
            settled_idx = i + 1
            break
    settling_time = (settled_idx if settled_idx else 0) * DT

    # Overshoot
    if abs(steady) > 1e-6:
        overshoot_pct = max(0, (states[:, channel].max() - steady) / abs(steady) * 100)
    else:
        overshoot_pct = 0.0

    # RMSE
    tracking_rmse = np.sqrt(np.mean(errors**2))

    # Control effort (RMS input)
    ctrl_effort = np.sqrt(np.mean(inputs[:, channel % 3]**2))

    return {
        "settling_s":     round(settling_time, 3),
        "overshoot_pct":  round(overshoot_pct, 1),
        "tracking_rmse":  round(float(tracking_rmse), 6),
        "ctrl_effort":    round(float(ctrl_effort), 6),
    }


# ─────────────────────────────────────────
#  Plot helpers
# ─────────────────────────────────────────
COLORS = {
    "Surrogate-MPC":   "#2563EB",
    "Linearised-MPC":  "#10B981",
    "PID":             "#F59E0B",
    "reference":       "#EF4444",
}

def plot_comparison(time_axis, trajs, refs, channel, ylabel, title, filename):
    """Generic comparison plot for one channel."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, refs[:, channel], "--",
            color=COLORS["reference"], linewidth=1.5, label="Reference", zorder=5)
    for name, traj in trajs.items():
        ax.plot(time_axis, traj[:, channel],
                color=COLORS[name], linewidth=2.0, label=name)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{filename}", dpi=150)
    plt.close()
    print(f"Saved: plots/{filename}")


# ─────────────────────────────────────────
#  Experiments
# ─────────────────────────────────────────
def experiment_1_step_response(controllers):
    """Roll step: 0 → 20° (0.349 rad)"""
    print("\n[Exp 1] Step Response: Roll 0 → 20°")
    T = 100          # 5 seconds total
    x0 = np.zeros(6)
    ref_angle = np.deg2rad(20.0)   # 0.349 rad

    def ref_fn(t):
        r = np.zeros(6)
        r[0] = ref_angle   # step at t=0, no delay
        return r

    results = {}
    metrics = {}
    for name, ctrl in controllers.items():
        states, inputs, _ = run_closed_loop(ctrl, x0, ref_fn, T)
        results[name] = states
        ref_states = np.array([ref_fn(t) for t in range(T + 1)])
        metrics[name] = compute_metrics(states, ref_states, inputs, channel=0)
        print(f"  {name:16s}: settling={metrics[name]['settling_s']:.2f}s  "
              f"overshoot={metrics[name]['overshoot_pct']:.1f}%  "
              f"RMSE={metrics[name]['tracking_rmse']:.4f} rad")

    ref_states = np.array([ref_fn(t) for t in range(T + 1)])
    time_axis  = np.arange(T + 1) * DT
    plot_comparison(time_axis, results, ref_states, 0,
                    "Roll angle φ (rad)", "Exp 1: Roll Step Response (0 → 20°)",
                    "exp1_step_response.png")
    return metrics

def experiment_2_sine_tracking(controllers):
    """Sinusoidal roll tracking: 15°·sin(0.5π·t)"""
    print("\n[Exp 2] Sine Tracking: φ_ref = 15° sin(0.5π·t)")
    T  = 200
    x0 = np.zeros(6)
    amp = np.deg2rad(15.0)
    omega = 0.5 * np.pi   # rad/s

    def ref_fn(t):
        r    = np.zeros(6)
        r[0] = amp * np.sin(omega * t * DT)
        return r

    results = {}
    metrics = {}
    for name, ctrl in controllers.items():
        states, inputs, _ = run_closed_loop(ctrl, x0, ref_fn, T)
        results[name] = states
        ref_states = np.array([ref_fn(t) for t in range(T + 1)])
        metrics[name] = compute_metrics(states, ref_states, inputs, channel=0)
        print(f"  {name:16s}: RMSE={metrics[name]['tracking_rmse']:.4f} rad")

    ref_states = np.array([ref_fn(t) for t in range(T + 1)])
    time_axis  = np.arange(T + 1) * DT
    plot_comparison(time_axis, results, ref_states, 0,
                    "Roll angle φ (rad)", "Exp 2: Sinusoidal Roll Tracking",
                    "exp2_sine_tracking.png")
    return metrics


def experiment_3_disturbance(controllers):
    """Impulse disturbance at t=3s while holding φ=0."""
    print("\n[Exp 3] Disturbance Rejection (impulse at t=3s)")
    T  = 160
    x0 = np.zeros(6)

    def ref_fn(t): return np.zeros(6)

    def disturb_fn(t, u):
        # Apply impulse disturbance at t=60 (3s) for 2 steps
        if 60 <= t < 62:
            u = u + np.array([0.008, 0.0, 0.0])
        return u

    results = {}
    metrics = {}
    for name, ctrl in controllers.items():
        states, inputs, _ = run_closed_loop(ctrl, x0, ref_fn, T,
                                             disturb_fn=disturb_fn)
        results[name] = states
        ref_states = np.zeros((T + 1, 6))
        metrics[name] = {
            "max_deviation": round(float(np.abs(states[:, 0]).max()), 5),
            "recovery_s":    round(float(np.argmax(np.abs(states[:, 0]) < 0.01)
                                         * DT - 3.0), 3),
        }
        print(f"  {name:16s}: max_dev={metrics[name]['max_deviation']:.4f} rad  "
              f"recovery≈{metrics[name]['recovery_s']:.2f}s")

    ref_states = np.zeros((T + 1, 6))
    time_axis  = np.arange(T + 1) * DT
    plot_comparison(time_axis, results, ref_states, 0,
                    "Roll angle φ (rad)", "Exp 3: Disturbance Rejection",
                    "exp3_disturbance.png")
    return metrics


def experiment_4_multiaxis(controllers):
    """Simultaneous roll + pitch + yaw step commands."""
    print("\n[Exp 4] Multi-axis simultaneous step")
    T  = 150
    x0 = np.zeros(6)

    def ref_fn(t):
        r = np.zeros(6)
        if t >= 5:
            r[0] = np.deg2rad(15)
            r[1] = np.deg2rad(10)
            r[2] = np.deg2rad(20)
        return r

    results = {}
    metrics = {}
    for name, ctrl in controllers.items():
        states, inputs, _ = run_closed_loop(ctrl, x0, ref_fn, T)
        results[name] = states
        ref_states = np.array([ref_fn(t) for t in range(T + 1)])
        m = {}
        for ch, cname in enumerate(["phi", "theta", "psi"]):
            m[cname + "_rmse"] = round(
                float(np.sqrt(np.mean((states[:, ch] - ref_states[:, ch])**2))), 5)
        metrics[name] = m
        print(f"  {name:16s}: φ-RMSE={m['phi_rmse']:.4f}  "
              f"θ-RMSE={m['theta_rmse']:.4f}  ψ-RMSE={m['psi_rmse']:.4f}")

    time_axis = np.arange(T + 1) * DT
    ref_states = np.array([ref_fn(t) for t in range(T + 1)])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    axis_labels = ["φ (roll)", "θ (pitch)", "ψ (yaw)"]
    for ax_i, (ax, label) in enumerate(zip(axes, axis_labels)):
        ax.plot(time_axis, ref_states[:, ax_i], "--",
                color=COLORS["reference"], label="Reference", linewidth=1.5)
        for name, traj in results.items():
            ax.plot(time_axis, traj[:, ax_i],
                    color=COLORS[name], linewidth=2.0, label=name)
        ax.set_title(label); ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Exp 4: Multi-axis Simultaneous Step", fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/exp4_multiaxis.png", dpi=150)
    plt.close()
    print("Saved: plots/exp4_multiaxis.png")
    return metrics


def experiment_5_model_mismatch(smpc):
    """Test Surrogate-MPC robustness when plant Ixx changes ±30%."""
    print("\n[Exp 5] Model Mismatch Robustness (Ixx varies ±30%)")
    T  = 120
    x0 = np.zeros(6)
    ref_angle = np.deg2rad(20)

    def ref_fn(t):
        r = np.zeros(6)
        r[0] = ref_angle if t >= 5 else 0.0
        return r

    variations = {
        "Ixx nominal": 1.00,
        "Ixx -30%":    0.70,
        "Ixx +30%":    1.30,
    }
    colors_mm = {"Ixx nominal": "#2563EB", "Ixx -30%": "#F59E0B", "Ixx +30%": "#EF4444"}

    results = {}
    metrics = {}
    for label, factor in variations.items():
        mismatch_params = PARAMS.copy()
        mismatch_params["Ixx"] = PARAMS["Ixx"] * factor
        states, inputs, _ = run_closed_loop(smpc, x0, ref_fn, T,
                                             plant_params=mismatch_params)
        results[label] = states
        ref_states = np.array([ref_fn(t) for t in range(T + 1)])
        metrics[label] = compute_metrics(states, ref_states, inputs, channel=0)
        print(f"  {label:15s}: settling={metrics[label]['settling_s']:.2f}s  "
              f"RMSE={metrics[label]['tracking_rmse']:.4f}")

    time_axis = np.arange(T + 1) * DT
    ref_states = np.array([ref_fn(t) for t in range(T + 1)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, ref_states[:, 0], "--",
            color="black", linewidth=1.5, label="Reference")
    for label, traj in results.items():
        ax.plot(time_axis, traj[:, 0],
                color=colors_mm[label], linewidth=2.0, label=label)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Roll φ (rad)")
    ax.set_title("Exp 5: Surrogate-MPC Robustness to Ixx Mismatch")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/exp5_mismatch.png", dpi=150)
    plt.close()
    print("Saved: plots/exp5_mismatch.png")
    return metrics


# ─────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────
def save_summary_table(step_metrics, sine_metrics, solve_times):
    rows = []
    for name in ["Surrogate-MPC", "Linearised-MPC", "PID"]:
        rows.append({
            "Controller":       name,
            "Step RMSE (rad)":  step_metrics[name]["tracking_rmse"],
            "Step Settling (s)": step_metrics[name]["settling_s"],
            "Step Overshoot (%)": step_metrics[name]["overshoot_pct"],
            "Sine RMSE (rad)":  sine_metrics[name]["tracking_rmse"],
            "Avg Solve (ms)":   round(float(np.mean(solve_times[name])), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/summary_table.csv", index=False)
    print("\n── Paper Summary Table ──")
    print(df.to_string(index=False))
    print("\nSaved: results/summary_table.csv")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Running All Closed-Loop Experiments")
    print("=" * 60)

    # Initialise controllers
    print("\nInitialising controllers...")
    smpc  = SurrogateMPC(N=10)
    lmpc  = LinearisedMPC(N=10)
    pid   = PIDController()

    controllers = {
        "Surrogate-MPC":  smpc,
        "Linearised-MPC": lmpc,
        "PID":            pid,
    }

    # Run experiments
    step_metrics = experiment_1_step_response(controllers)
    sine_metrics = experiment_2_sine_tracking(controllers)
    dist_metrics = experiment_3_disturbance(controllers)
    multi_metrics = experiment_4_multiaxis(controllers)
    mismatch_metrics = experiment_5_model_mismatch(smpc)

    # Collect solve times (run 30-step simulation purely for timing)
    print("\nBenchmarking solve times...")
    solve_times = {}
    x0 = np.zeros(6); ref = np.array([0.2, 0, 0, 0, 0, 0])
    for name, ctrl in controllers.items():
        times = []
        state = x0.copy()
        for _ in range(30):
            t0 = time.perf_counter()
            if hasattr(ctrl, "solve"):
                ctrl.solve(state, ref)
            else:
                ctrl.compute(state, ref)
            times.append((time.perf_counter() - t0) * 1000)
        solve_times[name] = times
        print(f"  {name:16s}: {np.mean(times):.2f} ± {np.std(times):.2f} ms")

    save_summary_table(step_metrics, sine_metrics, solve_times)

    print("\n" + "=" * 60)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  plots/exp1_step_response.png")
    print("  plots/exp2_sine_tracking.png")
    print("  plots/exp3_disturbance.png")
    print("  plots/exp4_multiaxis.png")
    print("  plots/exp5_mismatch.png")
    print("  results/summary_table.csv")