"""
quadrotor_simulator.py
======================
Ground-truth quadrotor attitude dynamics using Newton-Euler equations.
Uses Crazyflie 2.0 physical parameters (standard benchmark platform).

State:  x = [phi, theta, psi, p, q, r]   (angles in rad, rates in rad/s)
Input:  u = [tau_phi, tau_theta, tau_psi] (torques in N·m)
"""

import numpy as np
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────
#  Physical parameters (Crazyflie 2.0)
# ─────────────────────────────────────────
PARAMS = {
    "Ixx": 8.1e-3,     # kg·m²  roll moment of inertia
    "Iyy": 8.1e-3,     # kg·m²  pitch moment of inertia
    "Izz": 14.2e-3,    # kg·m²  yaw moment of inertia
    "mass": 0.5,       # kg
    "arm": 0.17,       # m   arm length
    "dt": 0.05,        # s   timestep
}


def quadrotor_dynamics(t, state, u, params=PARAMS):
    """
    Newton-Euler rotational dynamics.

    Parameters
    ----------
    t      : float   — current time (not used, required by solve_ivp)
    state  : array   — [phi, theta, psi, p, q, r]
    u      : array   — [tau_phi, tau_theta, tau_psi]
    params : dict    — physical parameters

    Returns
    -------
    dstate : array   — time derivatives [phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]
    """
    phi, theta, psi, p, q, r = state
    tau_phi, tau_theta, tau_psi = u

    Ixx = params["Ixx"]
    Iyy = params["Iyy"]
    Izz = params["Izz"]

    # Kinematic equations: angle rates
    phi_dot   = p
    theta_dot = q
    psi_dot   = r

    # Newton-Euler dynamic equations: angular acceleration
    p_dot = (Iyy - Izz) / Ixx * q * r  +  tau_phi   / Ixx
    q_dot = (Izz - Ixx) / Iyy * p * r  +  tau_theta / Iyy
    r_dot = (Ixx - Iyy) / Izz * p * q  +  tau_psi   / Izz

    return [phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]


def simulate_step(state, u, dt=PARAMS["dt"], params=PARAMS):
    """
    Simulate one timestep forward using RK45 integration.

    Parameters
    ----------
    state : array-like (6,)  — current state
    u     : array-like (3,)  — control input (torques)
    dt    : float            — timestep in seconds

    Returns
    -------
    next_state : np.ndarray (6,)
    """
    sol = solve_ivp(
        fun=quadrotor_dynamics,
        t_span=[0.0, dt],
        y0=np.array(state, dtype=float),
        args=(u, params),
        method="RK45",
        rtol=1e-6,
        atol=1e-8,
        dense_output=False,
    )
    return sol.y[:, -1]

def delta_linear(XU):
    """Exact Euler hover-linearised one-step delta for rows [state(6), input(3)].
    Physics base of the residual surrogate; matches LinearisedMPC's A,B."""
    XU = np.atleast_2d(np.asarray(XU, dtype=float))
    x, u = XU[:, :6], XU[:, 6:]
    dt = PARAMS["dt"]
    d = np.zeros((len(XU), 6))
    d[:, 0] = dt * x[:, 3]; d[:, 1] = dt * x[:, 4]; d[:, 2] = dt * x[:, 5]
    d[:, 3] = dt * u[:, 0] / PARAMS["Ixx"]
    d[:, 4] = dt * u[:, 1] / PARAMS["Iyy"]
    d[:, 5] = dt * u[:, 2] / PARAMS["Izz"]
    return d


def simulate_trajectory(x0, u_sequence, dt=PARAMS["dt"], params=PARAMS):
    """
    Simulate a full trajectory given an initial state and sequence of inputs.

    Parameters
    ----------
    x0         : array (6,)    — initial state
    u_sequence : array (T, 3)  — sequence of T control inputs
    dt         : float         — timestep

    Returns
    -------
    states : np.ndarray (T+1, 6)  — state trajectory including x0
    """
    T = len(u_sequence)
    states = np.zeros((T + 1, 6))
    states[0] = x0
    for t in range(T):
        states[t + 1] = simulate_step(states[t], u_sequence[t], dt, params)
    return states


# ─────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Quadrotor Simulator — Sanity Check")
    print("=" * 50)

    # Start at hover (all zeros) with a small roll torque
    x0 = np.zeros(6)
    u  = np.array([0.005, 0.0, 0.0])   # small roll torque

    x1 = simulate_step(x0, u)
    print(f"\nInitial state : {x0}")
    print(f"Control input : {u}")
    print(f"Next state    : {x1.round(6)}")
    print(f"\nExpected: phi and p should increase slightly (roll motion)")

    # Simulate 2-second trajectory
    T = 40   # 40 steps × 0.05 s = 2 s
    u_seq = np.tile(u, (T, 1))
    traj = simulate_trajectory(x0, u_seq)
    print(f"\nTrajectory shape: {traj.shape}  (41 timesteps × 6 states)")
    print(f"Final phi (roll): {traj[-1, 0]:.4f} rad  ({np.degrees(traj[-1, 0]):.2f} deg)")
    print("\n✓ Simulator working correctly")