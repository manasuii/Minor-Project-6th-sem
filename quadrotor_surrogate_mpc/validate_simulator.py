"""
validate_simulator.py
=====================
Validation step for Objective O1 (the proposal's flowchart node
"Validate simulator: Energy conservation & ODE baseline timing").

Two independent checks:

  1. Conservation test (physical correctness)
     For TORQUE-FREE rigid-body rotation (u = 0), the Newton-Euler attitude
     dynamics must conserve two invariants:
        * rotational kinetic energy   T   = 1/2 (Ixx p^2 + Iyy q^2 + Izz r^2)
        * angular-momentum magnitude  |H| = sqrt((Ixx p)^2 + (Iyy q)^2 + (Izz r)^2)
     We integrate several random spinning initial states and confirm the
     relative drift of T and |H| stays below tight tolerances.  This is the
     standard polhode/energy-ellipsoid invariance and verifies the integrator.

  2. ODE baseline timing
     Measures wall-clock cost of one RK45 prediction step, substantiating the
     proposal's 300-800 us/call figure and motivating the surrogate.

A note on dimensionality
------------------------
The state is x = [phi, theta, psi, p, q, r] in R^6 — three rotational degrees
of freedom (roll/pitch/yaw) described by six state variables.  Translational
position/velocity are intentionally out of scope (attitude-stabilisation
problem).  "6-DOF" in the proposal text refers to this 6-state attitude model.
"""

import time

import numpy as np

from quadrotor_simulator import simulate_step, simulate_trajectory, PARAMS


def kinetic_energy(state, params=PARAMS):
    p, q, r = state[3], state[4], state[5]
    return 0.5 * (params["Ixx"] * p**2 + params["Iyy"] * q**2 + params["Izz"] * r**2)


def angular_momentum_mag(state, params=PARAMS):
    p, q, r = state[3], state[4], state[5]
    return np.sqrt((params["Ixx"] * p)**2 + (params["Iyy"] * q)**2 + (params["Izz"] * r)**2)


def conservation_test(n_cases=8, T_steps=40, seed=42,
                      tol_energy=1e-4, tol_momentum=1e-4):
    """Torque-free integration must conserve T and |H|."""
    print("\n── Conservation test (torque-free rigid-body rotation) ──")
    rng = np.random.default_rng(seed)
    u_zero = np.zeros(3)
    worst_E, worst_H = 0.0, 0.0

    for case in range(n_cases):
        x0 = np.concatenate([
            rng.uniform(-0.5, 0.5, 3),      # arbitrary initial angles
            rng.uniform(-2.0, 2.0, 3),      # spinning body rates
        ])
        traj = simulate_trajectory(x0, np.tile(u_zero, (T_steps, 1)))

        E = np.array([kinetic_energy(s) for s in traj])
        H = np.array([angular_momentum_mag(s) for s in traj])
        dE = np.max(np.abs(E - E[0])) / (abs(E[0]) + 1e-12)
        dH = np.max(np.abs(H - H[0])) / (abs(H[0]) + 1e-12)
        worst_E = max(worst_E, dE)
        worst_H = max(worst_H, dH)

    print(f"  cases tested            : {n_cases}  (×{T_steps} steps, {T_steps*PARAMS['dt']:.1f}s)")
    print(f"  worst energy drift      : {worst_E:.2e}   (tol {tol_energy:.0e})")
    print(f"  worst |H| drift         : {worst_H:.2e}   (tol {tol_momentum:.0e})")
    ok = (worst_E < tol_energy) and (worst_H < tol_momentum)
    print(f"  result                  : {'PASS' if ok else 'FAIL'}")
    return ok


def directional_sanity():
    """A positive roll torque must increase roll angle and roll rate."""
    print("\n── Directional sanity ──")
    x1 = simulate_step(np.zeros(6), np.array([0.005, 0.0, 0.0]))
    ok = (x1[0] > 0) and (x1[3] > 0)
    print(f"  +roll torque -> phi={x1[0]:.3e} rad, p={x1[3]:.3e} rad/s : "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def timing_benchmark(n_trials=2000):
    """Wall-clock cost of one RK45 prediction step."""
    print("\n── ODE baseline timing (RK45 prediction step) ──")
    rng = np.random.default_rng(0)
    states = rng.uniform(-1, 1, (n_trials, 6))
    us = rng.uniform(-0.01, 0.01, (n_trials, 3))
    t0 = time.perf_counter()
    for i in range(n_trials):
        simulate_step(states[i], us[i])
    per_call_us = (time.perf_counter() - t0) / n_trials * 1e6
    print(f"  trials                  : {n_trials}")
    print(f"  mean cost per call      : {per_call_us:.1f} us")
    print(f"  (proposal cites ~300-800 us; an N-step MPC horizon pays this "
          f"per node, per SQP iteration)")
    return per_call_us


if __name__ == "__main__":
    print("=" * 60)
    print("Simulator Validation — Quadrotor Attitude Dynamics")
    print("=" * 60)
    print("\nPhysical parameters (Crazyflie 2.0):")
    for k in ["Ixx", "Iyy", "Izz", "dt"]:
        print(f"  {k:4s} = {PARAMS[k]}")

    ok1 = directional_sanity()
    ok2 = conservation_test()
    timing_benchmark()

    print("\n" + "=" * 60)
    print(f"✓ Simulator validation {'PASSED' if (ok1 and ok2) else 'FAILED'}")
    print("=" * 60)
