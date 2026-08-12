"""
baseline_controllers.py
=======================
Baseline controllers for benchmarking Surrogate-MPC (proposal Tier-2):

  B1. PIDController   — three independent PID loops (roll, pitch, yaw)
  B2. LinearisedMPC   — MPC with hover-linearised dynamics
  B3. ODEMPCController — MPC with the *true* nonlinear ODE (RK4) inside the
                         optimisation loop. This is the accuracy upper bound:
                         it shares Surrogate-MPC's horizon, cost weights, and
                         constraints, differing ONLY in the prediction model.
                         It is also the slow reference that motivates the
                         surrogate (the speedup baseline).

All controllers use identical input constraints for a fair comparison.
"""

import time

import numpy as np
import casadi as ca

from quadrotor_simulator import PARAMS
# Share the EXACT cost/horizon used by Surrogate-MPC so B2/B3 isolate the model.
from mpc_surrogate import (
    N_HORIZON, Q_DIAG, R_DIAG, TERMINAL_MULT, U_MAX, U_MIN,
)

DT  = PARAMS["dt"]
Ixx, Iyy, Izz = PARAMS["Ixx"], PARAMS["Iyy"], PARAMS["Izz"]


# ═════════════════════════════════════════
#  B1 — PID
# ═════════════════════════════════════════
class PIDController:
    """
    Three independent PID controllers for roll, pitch, yaw (Ziegler-Nichols
    seed + anti-windup). State [phi,theta,psi,p,q,r] -> [tau_phi,tau_theta,tau_psi].
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
        error = reference[:3] - state[:3]
        kp, ki, kd = self.gains[:, 0], self.gains[:, 1], self.gains[:, 2]
        self._integral = np.clip(self._integral + error * self.dt, -0.5, 0.5)
        derivative = -state[3:6]                # measured rate; avoids derivative kick
        u = kp * error + ki * self._integral + kd * derivative
        return np.clip(u, U_MIN, U_MAX)


# ═════════════════════════════════════════
#  B2 — Linearised MPC
# ═════════════════════════════════════════
class LinearisedMPC:
    """
    MPC using a first-order Taylor linearisation of the attitude dynamics about
    the hover equilibrium (x = 0, u = 0). Same horizon/weights/constraints as
    Surrogate-MPC for a fair comparison.
    """

    def __init__(self, N=N_HORIZON):
        self.N = N
        self.DT = DT
        self.Q  = np.diag(Q_DIAG)
        self.R  = np.diag(R_DIAG)
        self.Qf = TERMINAL_MULT * self.Q
        self._build_linearised_model()
        self._build_ocp()
        self._u_prev = np.zeros((N, 3))

    def _build_linearised_model(self):
        dt = self.DT
        Ac = np.zeros((6, 6))
        Ac[0, 3] = Ac[1, 4] = Ac[2, 5] = 1.0    # angle rates
        Bc = np.zeros((6, 3))
        Bc[3, 0] = 1.0 / Ixx
        Bc[4, 1] = 1.0 / Iyy
        Bc[5, 2] = 1.0 / Izz
        self.A = np.eye(6) + dt * Ac             # Euler discretisation
        self.B = dt * Bc

    def _build_ocp(self):
        opti = ca.Opti()
        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)
        x0  = opti.parameter(6)
        ref = opti.parameter(6)
        opti.subject_to(X[:, 0] == x0)

        Q, R, Qf = ca.DM(self.Q), ca.DM(self.R), ca.DM(self.Qf)
        A, B = ca.DM(self.A), ca.DM(self.B)

        cost = 0.0
        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == A @ X[:, k] + B @ U[:, k])
            e = X[:, k] - ref
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[:, k].T, R, U[:, k]])
            opti.subject_to(opti.bounded(U_MIN, U[:, k], U_MAX))
        eN = X[:, self.N] - ref
        cost += ca.mtimes([eN.T, Qf, eN])
        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0})

        self._opti, self._X, self._U = opti, X, U
        self._x0, self._ref = x0, ref

    def solve(self, state, reference):
        self._opti.set_value(self._x0, state)
        self._opti.set_value(self._ref, reference)
        self._opti.set_initial(self._U, self._u_prev.T)
        t0 = time.perf_counter()
        try:
            sol = self._opti.solve()
            u_seq = sol.value(self._U).T
            ms = (time.perf_counter() - t0) * 1000
            self._u_prev = np.roll(u_seq, -1, axis=0)
            return u_seq[0], ms, True
        except Exception:
            return np.zeros(3), (time.perf_counter() - t0) * 1000, False


# ═════════════════════════════════════════
#  B3 — ODE-MPC (true nonlinear dynamics inside the loop; accuracy upper bound)
# ═════════════════════════════════════════
class ODEMPCController:
    """
    Receding-horizon MPC that integrates the *true* Newton-Euler attitude ODE
    inside the optimisation, using a fixed-step RK4 discretisation (a
    differentiable stand-in for the adaptive RK45 used to generate the data).

    Shares Surrogate-MPC's horizon, cost weights, and input constraints, so the
    only difference is the prediction model. Solved with IPOPT (fully nonlinear)
    — this is the slow, high-accuracy reference that motivates the surrogate.
    """

    def __init__(self, N=N_HORIZON, rk_substeps=4):
        self.N = N
        self.M = rk_substeps
        self._build()
        self._u_prev = np.zeros((N, 3))

    @staticmethod
    def _f(x, u):
        """Symbolic continuous-time attitude dynamics (matches quadrotor_dynamics)."""
        phi, theta, psi = x[0], x[1], x[2]
        p, q, r = x[3], x[4], x[5]
        tphi, tth, tpsi = u[0], u[1], u[2]
        return ca.vertcat(
            p, q, r,
            (Iyy - Izz) / Ixx * q * r + tphi / Ixx,
            (Izz - Ixx) / Iyy * p * r + tth / Iyy,
            (Ixx - Iyy) / Izz * p * q + tpsi / Izz,
        )

    def _rk4_step(self, x, u):
        h = DT / self.M
        for _ in range(self.M):
            k1 = self._f(x, u)
            k2 = self._f(x + 0.5 * h * k1, u)
            k3 = self._f(x + 0.5 * h * k2, u)
            k4 = self._f(x + h * k3, u)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _build(self):
        opti = ca.Opti()
        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)
        x0  = opti.parameter(6)
        ref = opti.parameter(6)
        opti.subject_to(X[:, 0] == x0)

        Q  = ca.DM(np.diag(Q_DIAG))
        R  = ca.DM(np.diag(R_DIAG))
        Qf = TERMINAL_MULT * Q

        cost = 0.0
        for k in range(self.N):
            opti.subject_to(X[:, k + 1] == self._rk4_step(X[:, k], U[:, k]))
            e = X[:, k] - ref
            cost += ca.mtimes([e.T, Q, e]) + ca.mtimes([U[:, k].T, R, U[:, k]])
            opti.subject_to(opti.bounded(U_MIN, U[:, k], U_MAX))
        eN = X[:, self.N] - ref
        cost += ca.mtimes([eN.T, Qf, eN])
        opti.minimize(cost)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0,
                              "ipopt.max_iter": 100})

        self._opti, self._X, self._U = opti, X, U
        self._x0, self._ref = x0, ref

    def solve(self, state, reference):
        self._opti.set_value(self._x0, state)
        self._opti.set_value(self._ref, reference)
        self._opti.set_initial(self._U, self._u_prev.T)
        t0 = time.perf_counter()
        try:
            sol = self._opti.solve()
            u_seq = sol.value(self._U).T
            ms = (time.perf_counter() - t0) * 1000
            self._u_prev = np.roll(u_seq, -1, axis=0)
            return u_seq[0], ms, True
        except Exception:
            return np.zeros(3), (time.perf_counter() - t0) * 1000, False


# ═════════════════════════════════════════
#  Quick test
# ═════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Controllers — Quick Test")
    print("=" * 60)
    state = np.zeros(6)
    ref = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    pid = PIDController()
    print(f"\nPID output      : {pid.compute(state, ref)}")

    lmpc = LinearisedMPC()
    for _ in range(2):                    # warmup (first IPOPT call initialises)
        lmpc.solve(state, ref)
    u, t, ok = lmpc.solve(state, ref)
    print(f"Lin-MPC output  : {u}  ({t:.1f} ms, ok={ok})")

    ode = ODEMPCController()
    # warmup (first IPOPT call compiles/initialises)
    for _ in range(2):
        ode.solve(state, ref)
    u, t, ok = ode.solve(state, ref)
    print(f"ODE-MPC output  : {u}  ({t:.1f} ms, ok={ok})")

    print("\n✓ Baseline controllers working")
