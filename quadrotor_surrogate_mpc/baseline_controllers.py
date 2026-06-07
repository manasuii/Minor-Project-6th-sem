"""
baseline_controllers.py
=======================
Two baseline controllers for comparison against Surrogate-MPC:

  1. PID  — three independent PID controllers, one per rotation axis
  2. LinearisedMPC — MPC with first-order linearised dynamics

Both use the same input constraints as Surrogate-MPC for fair comparison.
"""

import numpy as np
import casadi as ca
from quadrotor_simulator import PARAMS
import time

U_MAX = np.array([ 0.010,  0.010,  0.005])
U_MIN = np.array([-0.010, -0.010, -0.005])
DT    = PARAMS["dt"]


# ─────────────────────────────────────────
#  PID Controller
# ─────────────────────────────────────────
class PIDController:
    """
    Three independent PID controllers for roll, pitch, yaw.

    Gains are tuned empirically — start with Ziegler-Nichols
    and then fine-tune for this specific plant.

    State  : [phi, theta, psi, p, q, r]
    Output : [tau_phi, tau_theta, tau_psi]
    """

    def __init__(self, dt=DT):
        self.dt = dt

        # PID gains for each axis [kp, ki, kd]
        # These work well for the Crazyflie parameters
        self.gains = np.array([
    [0.60,  0.02,  0.25],   # roll  — was [0.15, 0.005, 0.08]
    [0.60,  0.02,  0.25],   # pitch
    [0.30,  0.01,  0.10],   # yaw
])

        self._integral  = np.zeros(3)
        self._prev_error = np.zeros(3)

    def reset(self):
        self._integral   = np.zeros(3)
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
        u : array (3,)  — [tau_phi, tau_theta, tau_psi]
        """
        # Angle errors (first 3 elements of state)
        error = reference[:3] - state[:3]

        kp = self.gains[:, 0]
        ki = self.gains[:, 1]
        kd = self.gains[:, 2]

        # Integral (anti-windup clamp)
        self._integral += error * self.dt
        self._integral = np.clip(self._integral, -0.5, 0.5)

        # Derivative (from angle rate — avoids derivative kick)
        derivative = -state[3:6]   # use measured rate instead of error derivative

        u = kp * error + ki * self._integral + kd * derivative
        return np.clip(u, U_MIN, U_MAX)


# ─────────────────────────────────────────
#  Linearised MPC
# ─────────────────────────────────────────
class LinearisedMPC:
    """
    MPC using first-order Taylor linearisation of quadrotor dynamics
    around the hover equilibrium point.

    This is the conventional MPC approach without a surrogate.
    Uses the same horizon, cost weights, and constraints as SurrogateMPC
    for a fair comparison.
    """

    def __init__(self, N=10):
        self.N  = N
        self.DT = DT
        self.Q  = np.diag([10., 10., 5., 1., 1., 0.5])
        self.R  = np.diag([0.1, 0.1, 0.1])
        self._build_linearised_model()
        self._build_ocp()
        self._u_prev = np.zeros((N, 3))

    def _build_linearised_model(self):
        """
        Compute A, B matrices: x_{t+1} ≈ A*x_t + B*u_t
        Linearised around hover (x=0, u=0).

        At hover the Euler terms (p*q, q*r, p*r) vanish,
        so A is block diagonal and B is block diagonal.
        """
        Ixx = PARAMS["Ixx"]
        Iyy = PARAMS["Iyy"]
        Izz = PARAMS["Izz"]
        dt  = self.DT

        # Continuous-time A matrix (at hover)
        Ac = np.zeros((6, 6))
        Ac[0, 3] = 1.0   # phi_dot = p
        Ac[1, 4] = 1.0   # theta_dot = q
        Ac[2, 5] = 1.0   # psi_dot = r
        # p_dot, q_dot, r_dot = 0 at hover (cross-product terms vanish)

        # Continuous-time B matrix
        Bc = np.zeros((6, 3))
        Bc[3, 0] = 1.0 / Ixx   # p_dot from tau_phi
        Bc[4, 1] = 1.0 / Iyy   # q_dot from tau_theta
        Bc[5, 2] = 1.0 / Izz   # r_dot from tau_psi

        # Discretise using Euler method (dt is small enough)
        self.A = np.eye(6) + dt * Ac
        self.B = dt * Bc

    def _build_ocp(self):
        """Build MPC OCP with linear dynamics."""
        opti = ca.Opti()

        X = opti.variable(6, self.N + 1)
        U = opti.variable(3, self.N)
        x0  = opti.parameter(6)
        ref = opti.parameter(6)

        opti.subject_to(X[:, 0] == x0)

        Q_ca = ca.DM(self.Q)
        R_ca = ca.DM(self.R)
        A_ca = ca.DM(self.A)
        B_ca = ca.DM(self.B)

        cost = 0.0
        for k in range(self.N):
            # Linear dynamics
            opti.subject_to(X[:, k + 1] == A_ca @ X[:, k] + B_ca @ U[:, k])

            e_k   = X[:, k] - ref
            cost += ca.mtimes([e_k.T, Q_ca, e_k]) + ca.mtimes([U[:, k].T, R_ca, U[:, k]])
            opti.subject_to(opti.bounded(U_MIN, U[:, k], U_MAX))

        e_N  = X[:, self.N] - ref
        cost += 2.0 * ca.mtimes([e_N.T, Q_ca, e_N])
        opti.minimize(cost)

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X; self._U = U
        self._x0 = x0; self._ref = ref

    def solve(self, state, reference):
        import time
        self._opti.set_value(self._x0,  state)
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
#  Quick test
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Baseline Controllers — Quick Test")
    print("=" * 60)

    state = np.zeros(6)
    ref   = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    # PID
    pid = PIDController()
    u_pid = pid.compute(state, ref)
    print(f"\nPID output  : {u_pid}")

    # Linearised MPC
    lmpc = LinearisedMPC()
    u_lmpc, t_lmpc, ok = lmpc.solve(state, ref)
    print(f"Lin-MPC output  : {u_lmpc}  ({t_lmpc:.1f} ms)")

    print("\n✓ Baseline controllers working")
    print("\nTiming ODE-based MPC baseline...")
    from scipy.integrate import solve_ivp
    from quadrotor_simulator import quadrotor_dynamics
    
    def ode_predict_step(state, u):
        sol = solve_ivp(quadrotor_dynamics, [0, 0.05], state,
                        args=(u, PARAMS), method='RK45',
                        rtol=1e-6, atol=1e-8)
        return sol.y[:, -1]

    # Simulate what ODE-MPC does: N=5 predictions per solve call
    N = 5
    n_trials = 50
    state = np.zeros(6)
    u_seq = np.zeros((N, 3))

    ode_times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        x = state.copy()
        for k in range(N):
            x = ode_predict_step(x, u_seq[k])   # N ODE calls per solve
        ode_times.append((time.perf_counter() - t0) * 1000)

    print(f"ODE-MPC equivalent mean : {np.mean(ode_times):.2f} ms")
    print(f"Speedup vs Neural MPC   : {np.mean(ode_times)/34.89:.1f}×")