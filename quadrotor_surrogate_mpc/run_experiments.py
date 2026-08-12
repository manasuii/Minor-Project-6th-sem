"""
run_experiments.py
==================
Closed-loop benchmarking (proposal Tier-2). Six tracking scenarios, four
controllers: Surrogate-MPC (proposed) vs PID, Linearised-MPC, and ODE-MPC
(true nonlinear dynamics — the accuracy/timing upper bound).

Scenarios:
  1. Roll step response       (0 -> 20 deg)
  2. Sinusoidal roll tracking  (15 deg sine)
  3. Disturbance rejection      (impulse torque at t = 3 s)
  4. Multi-axis step            (simultaneous roll + pitch + yaw)
  5. Yaw step response          (0 -> 30 deg; isolates Izz / lower torque authority)
  6. Parametric plant mismatch  (Ixx +/-30%) — run for ALL controllers

Outputs:
  plots/exp1_step_response.png ... plots/exp6_mismatch.png
  results/summary_table.csv
  results/realtime_feasibility.csv
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from quadrotor_simulator import simulate_step, PARAMS
from mpc_surrogate import SurrogateMPC, N_HORIZON
from baseline_controllers import PIDController, LinearisedMPC, ODEMPCController

os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

DT = PARAMS["dt"]
RT_BUDGET_MS = 50.0     # proposal real-time constraint: <= 50 ms (>= 20 Hz)

COLORS = {
    "Surrogate-MPC":  "#2563EB",
    "ODE-MPC":        "#7C3AED",
    "Linearised-MPC": "#10B981",
    "PID":            "#F59E0B",
    "reference":      "#EF4444",
}
ORDER = ["Surrogate-MPC", "ODE-MPC", "Linearised-MPC", "PID"]


# ─────────────────────────────────────────
#  Closed-loop runner
# ─────────────────────────────────────────
def run_closed_loop(controller, x0, ref_fn, T_steps, disturb_fn=None, plant_params=None):
    params = plant_params if plant_params is not None else PARAMS
    states = np.zeros((T_steps + 1, 6))
    inputs = np.zeros((T_steps, 3))
    solve_times = []
    states[0] = x0
    if hasattr(controller, "reset"):
        controller.reset()

    for t in range(T_steps):
        ref, state = ref_fn(t), states[t]
        t0 = time.perf_counter()
        if hasattr(controller, "solve"):
            u, ms, _ = controller.solve(state, ref)
        else:
            u = controller.compute(state, ref)
            ms = (time.perf_counter() - t0) * 1000
        if disturb_fn is not None:
            u = disturb_fn(t, u)
        u = np.clip(u, [-0.010, -0.010, -0.005], [0.010, 0.010, 0.005])
        inputs[t] = u
        solve_times.append(ms)
        states[t + 1] = simulate_step(state, u, params=params)
    return states, inputs, solve_times


# ─────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────
def compute_metrics(states, ref_states, inputs, channel=0):
    errors = states[:, channel] - ref_states[:, channel]
    steady = ref_states[-1, channel]
    tol = 0.02 * abs(steady) if abs(steady) > 1e-6 else 0.005
    settled_idx = 0
    for i in range(len(errors) - 1, -1, -1):
        if abs(errors[i]) > tol:
            settled_idx = i + 1
            break
    settling_time = settled_idx * DT
    overshoot = (max(0, (states[:, channel].max() - steady) / abs(steady) * 100)
                 if abs(steady) > 1e-6 else 0.0)
    return {
        "settling_s":    round(settling_time, 3),
        "overshoot_pct": round(overshoot, 1),
        "tracking_rmse": round(float(np.sqrt(np.mean(errors ** 2))), 6),
        "ctrl_effort":   round(float(np.sqrt(np.mean(inputs[:, channel % 3] ** 2))), 6),
    }


def plot_comparison(time_axis, trajs, refs, channel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, refs[:, channel], "--", color=COLORS["reference"],
            linewidth=1.5, label="Reference", zorder=5)
    for name in ORDER:
        if name in trajs:
            ax.plot(time_axis, trajs[name][:, channel], color=COLORS[name],
                    linewidth=2.0, label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(f"plots/{filename}", dpi=150); plt.close()
    print(f"Saved: plots/{filename}")


# ─────────────────────────────────────────
#  Experiments 1-5
# ─────────────────────────────────────────
def experiment_step(controllers):
    print("\n[Exp 1] Roll step response: 0 -> 20 deg")
    T, x0, ang = 100, np.zeros(6), np.deg2rad(20.0)
    ref_fn = lambda t: np.array([ang, 0, 0, 0, 0, 0])
    results, metrics = {}, {}
    refs = np.array([ref_fn(t) for t in range(T + 1)])
    for name, c in controllers.items():
        st, inp, _ = run_closed_loop(c, x0, ref_fn, T)
        results[name] = st
        metrics[name] = compute_metrics(st, refs, inp, 0)
        print(f"  {name:16s}: settling={metrics[name]['settling_s']:.2f}s  "
              f"overshoot={metrics[name]['overshoot_pct']:.1f}%  "
              f"RMSE={metrics[name]['tracking_rmse']:.4f} rad")
    plot_comparison(np.arange(T + 1) * DT, results, refs, 0,
                    "Roll angle phi (rad)", "Exp 1: Roll Step (0 -> 20 deg)",
                    "exp1_step_response.png")
    return metrics


def experiment_sine(controllers):
    print("\n[Exp 2] Sinusoidal roll tracking: 15 deg sin(0.5*pi*t)")
    T, x0 = 200, np.zeros(6)
    amp, omega = np.deg2rad(15.0), 0.5 * np.pi
    ref_fn = lambda t: np.array([amp * np.sin(omega * t * DT), 0, 0, 0, 0, 0])
    results, metrics = {}, {}
    refs = np.array([ref_fn(t) for t in range(T + 1)])
    for name, c in controllers.items():
        st, inp, _ = run_closed_loop(c, x0, ref_fn, T)
        results[name] = st
        metrics[name] = compute_metrics(st, refs, inp, 0)
        print(f"  {name:16s}: RMSE={metrics[name]['tracking_rmse']:.4f} rad")
    plot_comparison(np.arange(T + 1) * DT, results, refs, 0,
                    "Roll angle phi (rad)", "Exp 2: Sinusoidal Roll Tracking",
                    "exp2_sine_tracking.png")
    return metrics


def experiment_disturbance(controllers):
    print("\n[Exp 3] Disturbance rejection (impulse at t = 3 s)")
    T, x0 = 160, np.zeros(6)
    ref_fn = lambda t: np.zeros(6)

    def disturb_fn(t, u):
        return u + np.array([0.008, 0.0, 0.0]) if 60 <= t < 62 else u

    results, metrics = {}, {}
    for name, c in controllers.items():
        st, inp, _ = run_closed_loop(c, x0, ref_fn, T, disturb_fn=disturb_fn)
        results[name] = st
        phi = np.abs(st[:, 0])
        rec = np.nan
        for i in range(62, len(phi)):
            if np.all(phi[i:] < 0.01):
                rec = (i - 62) * DT
                break
        metrics[name] = {"max_deviation": round(float(phi.max()), 5),
                         "recovery_s": round(float(rec), 3) if rec == rec else None}
        print(f"  {name:16s}: max_dev={metrics[name]['max_deviation']:.4f} rad  "
              f"recovery={metrics[name]['recovery_s']}s")
    plot_comparison(np.arange(T + 1) * DT, results, np.zeros((T + 1, 6)), 0,
                    "Roll angle phi (rad)", "Exp 3: Disturbance Rejection",
                    "exp3_disturbance.png")
    return metrics


def experiment_multiaxis(controllers):
    print("\n[Exp 4] Multi-axis simultaneous step")
    T, x0 = 150, np.zeros(6)

    def ref_fn(t):
        r = np.zeros(6)
        if t >= 5:
            r[0], r[1], r[2] = np.deg2rad(15), np.deg2rad(10), np.deg2rad(20)
        return r

    results, metrics = {}, {}
    refs = np.array([ref_fn(t) for t in range(T + 1)])
    for name, c in controllers.items():
        st, inp, _ = run_closed_loop(c, x0, ref_fn, T)
        results[name] = st
        m = {f"{cn}_rmse": round(float(np.sqrt(np.mean((st[:, ch] - refs[:, ch]) ** 2))), 5)
             for ch, cn in enumerate(["phi", "theta", "psi"])}
        metrics[name] = m
        print(f"  {name:16s}: phi-RMSE={m['phi_rmse']:.4f}  "
              f"theta-RMSE={m['theta_rmse']:.4f}  psi-RMSE={m['psi_rmse']:.4f}")

    t = np.arange(T + 1) * DT
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax_i, (ax, lbl) in enumerate(zip(axes, ["phi (roll)", "theta (pitch)", "psi (yaw)"])):
        ax.plot(t, refs[:, ax_i], "--", color=COLORS["reference"],
                label="Reference", linewidth=1.5)
        for name in ORDER:
            if name in results:
                ax.plot(t, results[name][:, ax_i], color=COLORS[name],
                        linewidth=2.0, label=name)
        ax.set_title(lbl); ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Exp 4: Multi-axis Simultaneous Step", fontsize=13)
    plt.tight_layout(); plt.savefig("plots/exp4_multiaxis.png", dpi=150); plt.close()
    print("Saved: plots/exp4_multiaxis.png")
    return metrics


def experiment_yaw_step(controllers):
    print("\n[Exp 5] Yaw step response: 0 -> 30 deg (isolates Izz axis)")
    T, x0, ang = 120, np.zeros(6), np.deg2rad(30.0)
    ref_fn = lambda t: np.array([0, 0, ang, 0, 0, 0])
    results, metrics = {}, {}
    refs = np.array([ref_fn(t) for t in range(T + 1)])
    for name, c in controllers.items():
        st, inp, _ = run_closed_loop(c, x0, ref_fn, T)
        results[name] = st
        metrics[name] = compute_metrics(st, refs, inp, 2)
        print(f"  {name:16s}: settling={metrics[name]['settling_s']:.2f}s  "
              f"overshoot={metrics[name]['overshoot_pct']:.1f}%  "
              f"RMSE={metrics[name]['tracking_rmse']:.4f} rad")
    plot_comparison(np.arange(T + 1) * DT, results, refs, 2,
                    "Yaw angle psi (rad)", "Exp 5: Yaw Step (0 -> 30 deg)",
                    "exp5_yaw_step.png")
    return metrics


def experiment_mismatch(controllers):
    """Parametric plant mismatch (Ixx +/-30%) — every controller."""
    print("\n[Exp 6] Parametric plant mismatch (Ixx +/-30%) — all controllers")
    T, x0, ang = 120, np.zeros(6), np.deg2rad(20.0)
    ref_fn = lambda t: np.array([ang if t >= 5 else 0.0, 0, 0, 0, 0, 0])
    refs = np.array([ref_fn(t) for t in range(T + 1)])
    variations = {"nominal": 1.00, "-30%": 0.70, "+30%": 1.30}

    metrics = {name: {} for name in controllers}
    nominal_traj = {}
    for name, c in controllers.items():
        for label, factor in variations.items():
            p = PARAMS.copy(); p["Ixx"] = PARAMS["Ixx"] * factor
            st, inp, _ = run_closed_loop(c, x0, ref_fn, T, plant_params=p)
            metrics[name][label] = compute_metrics(st, refs, inp, 0)
            if label == "nominal":
                nominal_traj[name] = st
        rng = [metrics[name][k]["tracking_rmse"] for k in variations]
        print(f"  {name:16s}: RMSE nominal/-30%/+30% = "
              f"{rng[0]:.4f} / {rng[1]:.4f} / {rng[2]:.4f}  "
              f"(spread {max(rng)-min(rng):.4f})")

    plot_comparison(np.arange(T + 1) * DT, nominal_traj, refs, 0,
                    "Roll angle phi (rad)",
                    "Exp 6: Roll Step under Plant Mismatch (nominal Ixx shown)",
                    "exp6_mismatch.png")
    return metrics


# ─────────────────────────────────────────
#  Summary + real-time feasibility
# ─────────────────────────────────────────
def benchmark_solve_times(controllers, n=40):
    print("\nBenchmarking solve times...")
    x0, ref = np.zeros(6), np.array([0.2, 0, 0, 0, 0, 0])
    out = {}
    for name, c in controllers.items():
        if hasattr(c, "solve"):           # warmup MPC solvers
            for _ in range(3):
                c.solve(x0, ref)
        times, state = [], x0.copy()
        for _ in range(n):
            t0 = time.perf_counter()
            if hasattr(c, "solve"):
                c.solve(state, ref)
            else:
                c.compute(state, ref)
            times.append((time.perf_counter() - t0) * 1000)
        out[name] = times
        print(f"  {name:16s}: {np.mean(times):.2f} +/- {np.std(times):.2f} ms")
    return out


def save_tables(step_m, sine_m, yaw_m, solve_times):
    rows = []
    for name in ORDER:
        rows.append({
            "Controller": name,
            "Step RMSE (rad)": step_m[name]["tracking_rmse"],
            "Step Settling (s)": step_m[name]["settling_s"],
            "Step Overshoot (%)": step_m[name]["overshoot_pct"],
            "Yaw RMSE (rad)": yaw_m[name]["tracking_rmse"],
            "Sine RMSE (rad)": sine_m[name]["tracking_rmse"],
            "Avg Solve (ms)": round(float(np.mean(solve_times[name])), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results/summary_table.csv", index=False)
    print("\n── Paper Summary Table ──")
    print(df.to_string(index=False))
    print("\nSaved: results/summary_table.csv")

    rt = []
    for name in ORDER:
        mean_ms = float(np.mean(solve_times[name]))
        rt.append({
            "Controller": name,
            "Avg Solve (ms)": round(mean_ms, 2),
            "Max Solve (ms)": round(float(np.max(solve_times[name])), 2),
            "Max Rate (Hz)": round(1000.0 / mean_ms, 1) if mean_ms > 0 else float("inf"),
            f"Real-time (<= {RT_BUDGET_MS:.0f} ms)": "YES" if mean_ms <= RT_BUDGET_MS else "NO",
        })
    df_rt = pd.DataFrame(rt)
    df_rt.to_csv("results/realtime_feasibility.csv", index=False)
    print("\n── Real-time Feasibility (proposal: <= 50 ms / >= 20 Hz) ──")
    print(df_rt.to_string(index=False))
    print("\nSaved: results/realtime_feasibility.csv")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Running All Closed-Loop Experiments (6 scenarios, 4 controllers)")
    print("=" * 60)

    print("\nInitialising controllers...")
    controllers = {
        "Surrogate-MPC":  SurrogateMPC(N=N_HORIZON),
        "ODE-MPC":        ODEMPCController(N=N_HORIZON),
        "Linearised-MPC": LinearisedMPC(N=N_HORIZON),
        "PID":            PIDController(),
    }

    step_m  = experiment_step(controllers)
    sine_m  = experiment_sine(controllers)
    _       = experiment_disturbance(controllers)
    _       = experiment_multiaxis(controllers)
    yaw_m   = experiment_yaw_step(controllers)
    _       = experiment_mismatch(controllers)

    solve_times = benchmark_solve_times(controllers)
    save_tables(step_m, sine_m, yaw_m, solve_times)

    print("\n" + "=" * 60)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    for f in ["exp1_step_response", "exp2_sine_tracking", "exp3_disturbance",
              "exp4_multiaxis", "exp5_yaw_step", "exp6_mismatch"]:
        print(f"  plots/{f}.png")
    print("  results/summary_table.csv")
    print("  results/realtime_feasibility.csv")
