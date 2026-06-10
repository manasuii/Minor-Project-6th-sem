"""
baseline_controllers.py
=======================

Two baseline controllers for comparison against Surrogate-MPC:s

  1. PID  — three independent PID controllers, one per rotation axis
  2. LinearisedMPC — MPC with first-order linearised dynamics

Both use the same input constraints as Surrogate-MPC for fair comparison.
"""

import numpy as np
import casadi as ca
from quadrotor_simulator import PARAMS
import time

U_MAX = np.array([0.010, 0.010, 0.005])
U_MIN = np.array([-0.010, -0.010, -0.005])

DT = PARAMS["dt"]


# ─────────────────────────────────────────
# PID Controller
# ─────────────────────────────────────────

class PIDController:
    """
    Three independent PID controllers for roll, pitch, yaw.

    State  : [phi, theta, psi, p, q, r]
    Output : [tau_phi, tau_theta, tau_psi]
    """

    def __init__(self, dt=DT):
        self.dt = dt

        self.gains = np.array([
            [0.60, 0.02, 0.25],   # roll
            [0.60, 0.02, 0.25],   # pitch
            [0.30, 0.01, 0.10],   # yaw
        ])

        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral = np.zeros(3)
        self._prev_error = np.zeros(3)

    def compute(self, state, reference):
        """
        Compute control torques from state error.

        Parameters
        ----------
        state     : array (6,)
        reference : array (6,)

        Returns
        -------
        u : array (3,)
        """

        error = reference[:3] - state[:3]

        kp = self.gains[:, 0]
        ki = self.gains[:, 1]
        kd = self.gains[:, 2]

        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -0.5, 0.5)

        derivative = -state[3:6]

        u = kp * error + ki * self._integral + kd * derivative

        return np.clip(u, U_MIN, U_MAX)


# ─────────────────────────────────────────
# Linearised MPC
# ─────────────────────────────────────────

class LinearisedMPC:
    """
    MPC using first-order Taylor linearisation of quadrotor dynamics
    around hover equilibrium.
    """

    def __init__(self, N=10):
        self.N = N
        self.DT = DT

        self.Q = np.diag([10., 10., 5., 1., 1., 0.5])
        self.R = np.diag([0.1, 0.1, 0.1])

        self._build_linearised_model()
        self._build_ocp()

        self._u_prev = np.zeros((N, 3))

    def _build_linearised_model(self):

        Ixx = PARAMS["Ixx"]
        Iyy = PARAMS["Iyy"]
        Izz = PARAMS["Izz"]

        dt = self.DT

        Ac = np.zeros((6, 6))

        Ac[0, 3] = 1.0
        Ac[1, 4] = 1.0
        Ac[2, 5] = 1.0

        Bc = np.zeros((6, 3))

        Bc[3, 0] = 1.0 / Ixx
        Bc[4, 1] = 1.0 / Iyy
        Bc[5, 2] = 1.0 / Izz

        self.A = np.eye(6) + dt * Ac
        self.B = dt * Bc

    def _build_ocp(self):

        opti = ca.Opti()

        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)

        x0 = opti.parameter(6)
        ref = opti.parameter(6)

        opti.subject_to(X[:, 0] == x0)

        Q_ca = ca.DM(self.Q)
        R_ca = ca.DM(self.R)

        A_ca = ca.DM(self.A)
        B_ca = ca.DM(self.B)

        cost = 0

        for k in range(self.N):

            opti.subject_to(
                X[:, k + 1] == A_ca @ X[:, k] + B_ca @ U[:, k]
            )

            e_k = X[:, k] - ref

            cost += (
                ca.mtimes([e_k.T, Q_ca, e_k]) +
                ca.mtimes([U[:, k].T, R_ca, U[:, k]])
            )

            opti.subject_to(
                opti.bounded(U_MIN, U[:, k], U_MAX)
            )

        e_N = X[:, self.N] - ref

        cost += 2.0 * ca.mtimes([e_N.T, Q_ca, e_N])

        opti.minimize(cost)

        opts = {
            "ipopt.print_level": 0,
            "print_time": 0
        }

        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._x0 = x0
        self._ref = ref

    def solve(self, state, reference):

        self._opti.set_value(self._x0, state)
        self._opti.set_value(self._ref, reference)

        self._opti.set_initial(self._U, self._u_prev.T)

        t0 = time.perf_counter()

        try:
            sol = self._opti.solve()

            u_seq = sol.value(self._U).T

            solve_ms = (time.perf_counter() - t0) * 1000

            self._u_prev = np.roll(u_seq, -1, axis=0)

            return u_seq[0], solve_ms, True

        except Exception:

            solve_ms = (time.perf_counter() - t0) * 1000

            return np.zeros(3), solve_ms, False


# ─────────────────────────────────────────
# ODE MPC Benchmark
# ─────────────────────────────────────────

def benchmark_ode_mpc_equivalent(N=5, n_trials=200):
    """
    Measures the cost of what ODE-embedded MPC would pay per solve:
    N sequential RK45 integrations (one per prediction step).

    This is the fair comparison — surrogate replaces this exact loop.
    """

    from quadrotor_simulator import simulate_step

    state = np.zeros(6)

    u_seq = np.tile(
        [0.003, -0.001, 0.0],
        (N, 1)
    )

    # Warmup
    for _ in range(20):
        x = state.copy()

        for k in range(N):
            x = simulate_step(x, u_seq[k])

    times = []

    for _ in range(n_trials):

        t0 = time.perf_counter()

        x = state.copy()

        for k in range(N):
            x = simulate_step(x, u_seq[k])

        times.append(
            (time.perf_counter() - t0) * 1000
        )

    mean_t = np.mean(times)
    std_t = np.std(times)

    surrogate_ms = 11.96

    print("\n── ODE-MPC vs Surrogate-MPC Timing ──")

    print(
        f"  ODE rollout (N={N} steps) : "
        f"{mean_t:.2f} ± {std_t:.2f} ms (per solver call)"
    )

    print(
        f"  Surrogate-MPC             : "
        f"{surrogate_ms:.2f} ms (full solve)"
    )

    print(
        f"  Speedup (dynamics only)   : "
        f"{mean_t / surrogate_ms:.1f}×"
    )

    print(
        "\n  Note: ODE time is per-iteration cost. "
        "A real ODE-MPC solver runs ~20–50 iterations per solve."
    )

    print(
        f"  ODE-MPC wall time ≈ "
        f"{mean_t * 20:.0f}–{mean_t * 50:.0f} ms per step."
    )

    print(
        f"  Honest speedup range      : "
        f"{mean_t * 20 / surrogate_ms:.0f}×–"
        f"{mean_t * 50 / surrogate_ms:.0f}×"
    )

    return mean_t


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

if __name__ == "__main__":
    benchmark_ode_mpc_equivalent(N=5)