"""
mpc_surrogate.py  (FAST VERSION — retuned with physics-informed base)
=====================================================================
Surrogate-Based MPC using SQP + OSQP solver instead of IPOPT.
SQP is 10-30x faster than IPOPT for small structured NLPs.
"""

import numpy as np
import casadi as ca
import torch
import os, time

from train_nn_surrogate import QuadrotorNNSurrogate
from quadrotor_simulator import PARAMS

# ─────────────────────────────────────────
#  MPC Parameters
# ─────────────────────────────────────────
N_HORIZON = 5           # prediction steps (0.25s lookahead)
DT        = PARAMS["dt"]
Ixx, Iyy, Izz = PARAMS["Ixx"], PARAMS["Iyy"], PARAMS["Izz"]

# Angle weights moderated from 200/200/100; rate (damping) weights raised.
Q_DIAG = [500.0, 500.0, 250.0,
           20.0,  20.0,  10.0]
R_DIAG = [0.005, 0.005, 0.005]
TERMINAL_MULT = 12.0
U_MAX = np.array([ 0.010,  0.010,  0.005])
U_MIN = np.array([-0.010, -0.010, -0.005])


# ─────────────────────────────────────────
#  Extract NN weights
# ─────────────────────────────────────────
def extract_nn_weights(model_path="models/nn_surrogate.pth",
                       cfg_path="models/nn_surrogate_cfg.npz"):
    model = QuadrotorNNSurrogate()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    weights, biases = [], []
    for name, param in model.named_parameters():
        arr = param.detach().numpy()
        if "weight" in name:
            weights.append(arr)
        elif "bias" in name:
            biases.append(arr)

    cfg = np.load(cfg_path)
    return weights, biases, cfg


# ─────────────────────────────────────────
#  Build CasADi symbolic NN (SiLU / Physics-embedded)
# ─────────────────────────────────────────
def build_casadi_nn(weights, biases, X_mean, X_std, Y_mean, Y_std,
                    zero_offset=True):
    """
    Build a CasADi Function mapping [x; u] (9,) -> delta (6,).
    Integrates the physics-informed linear base mapping back inside the model.
    """
    xu_sym    = ca.MX.sym("xu", 9)
    h         = (xu_sym - ca.DM(X_mean)) / ca.DM(X_std)

    for i, (W, b) in enumerate(zip(weights, biases)):
        h = ca.DM(W) @ h + ca.DM(b)
        if i < len(weights) - 1:
            # CasADi implementation of SiLU: x * sigmoid(x)
            h = h * (1.0 / (1.0 + ca.exp(-h)))

    corr  = h * ca.DM(Y_std) + ca.DM(Y_mean)          # NN nonlinear correction
    f_raw = ca.Function("f_corr", [xu_sym], [corr])
    
    if zero_offset:
        corr = corr - ca.DM(f_raw(ca.DM.zeros(9)))    # correction(0,0) = 0

    x_s, u_s = xu_sym[:6], xu_sym[6:]
    delta_lin = ca.vertcat(DT*x_s[3], DT*x_s[4], DT*x_s[5],
                           DT*u_s[0]/Ixx, DT*u_s[1]/Iyy, DT*u_s[2]/Izz)
                           
    return ca.Function("f_nn", [xu_sym], [delta_lin + corr])


# ─────────────────────────────────────────
#  Build the NLP once (compiled)
# ─────────────────────────────────────────
def build_nlp(nn_fn, N=N_HORIZON):
    """
    Build the MPC NLP directly using CasADi symbolics.
    Returns a compiled ca.Function for fast repeated solving.
    """
    nx, nu = 6, 3
    Q  = ca.DM(np.diag(Q_DIAG))
    R  = ca.DM(np.diag(R_DIAG))
    Qf = TERMINAL_MULT * Q

    # Decision variable: flat vector [u_0, u_1, ..., u_{N-1}]
    U_flat = ca.MX.sym("U", nu * N)
    x0_p   = ca.MX.sym("x0", nx)
    ref_p  = ca.MX.sym("ref", nx)

    cost = ca.MX(0)
    g    = []        # constraints (empty — bounds handled via lbx/ubx)
    x    = x0_p

    for k in range(N):
        u_k   = U_flat[k*nu : (k+1)*nu]
        xu_k  = ca.vertcat(x, u_k)
        delta = nn_fn(xu_k)
        x     = x + delta                           # residual update

        e_k   = x - ref_p
        cost += ca.mtimes(e_k.T, ca.mtimes(Q, e_k)) \
              + ca.mtimes(u_k.T, ca.mtimes(R, u_k))

    # Terminal cost (single clean multiplier via Qf)
    e_N   = x - ref_p
    cost += ca.mtimes(e_N.T, ca.mtimes(Qf, e_N))

    nlp = {"x": U_flat, "f": cost, "g": ca.vertcat(*g),
           "p": ca.vertcat(x0_p, ref_p)}

    # ── Solver: sqpmethod + osqp (fast for small structured NLPs) ──
    solver_opts = {
        "qpsol":        "osqp",
        "qpsol_options": {
            "osqp.verbose":       False,
            "osqp.eps_abs":       1e-4,
            "osqp.eps_rel":       1e-4,
            "osqp.max_iter":      1000,
            "osqp.warm_starting": True,
        },
        "print_header":    False,
        "print_iteration": False,
        "print_time":      False,
        "max_iter":        15,          # SQP outer iterations
        "tol_pr":          1e-4,
        "tol_du":          1e-4,
    }

    try:
        solver = ca.nlpsol("mpc_solver", "sqpmethod", nlp, solver_opts)
        print("  Using solver: SQP + OSQP")
    except Exception:
        print("  OSQP not found — falling back to IPOPT")
        ipopt_opts = {
            "ipopt.print_level":             0,
            "print_time":                    0,
            "ipopt.max_iter":                30,
            "ipopt.tol":                     1e-3,
            "ipopt.acceptable_tol":          1e-2,
            "ipopt.acceptable_iter":         3,
            "ipopt.hessian_approximation":   "limited-memory",
            "ipopt.warm_start_init_point":   "yes",
            "ipopt.mu_init":                 1e-2,
        }
        solver = ca.nlpsol("mpc_solver", "ipopt", nlp, ipopt_opts)
        print("  Using solver: IPOPT (L-BFGS)")

    # Input bounds
    lbu = np.tile(U_MIN, N)
    ubu = np.tile(U_MAX, N)

    return solver, lbu, ubu


# ─────────────────────────────────────────
#  MPC Class
# ─────────────────────────────────────────
class SurrogateMPC:
    def __init__(self, N=N_HORIZON):
        self.N  = N
        self.nx, self.nu = 6, 3

        print("Loading NN surrogate...")
        weights, biases, cfg = extract_nn_weights()
        self.nn_fn = build_casadi_nn(
            weights, biases,
            cfg["X_mean"], cfg["X_std"],
            cfg["Y_mean"], cfg["Y_std"],
            zero_offset=True,
        )

        print("Compiling MPC problem...")
        self.solver, self.lbu, self.ubu = build_nlp(self.nn_fn, N)

        # Warm-start buffer
        self._u0 = np.zeros(self.nu * N)
        print("MPC ready.\n")

    def solve(self, state, reference):
        p_val = np.concatenate([state, reference])

        t0  = time.perf_counter()
        sol = self.solver(
            x0=self._u0,
            p=p_val,
            lbx=self.lbu,
            ubx=self.ubu,
            lbg=[],
            ubg=[],
        )
        ms = (time.perf_counter() - t0) * 1000

        u_seq = np.array(sol["x"]).flatten()
        success = self.solver.stats()["success"]

        # Warm-start shift
        u_mat = u_seq.reshape(self.N, self.nu)
        self._u0 = np.roll(u_mat, -1, axis=0).flatten()

        return u_seq[:self.nu], ms, success


# ─────────────────────────────────────────
#  Test
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Surrogate MPC — Fast Solver Test")
    print("=" * 55)

    mpc = SurrogateMPC(N=N_HORIZON)

    state = np.zeros(6)
    ref   = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Warmup
    print("Warming up solver (3 calls)...")
    for _ in range(3):
        mpc.solve(state, ref)

    # Benchmark
    u_opt, ms, ok = mpc.solve(state, ref)
    print(f"\nOptimal input : tau_phi={u_opt[0]:.4f}  "
          f"tau_theta={u_opt[1]:.4f}  tau_psi={u_opt[2]:.4f} N·m")
    print(f"Solve time    : {ms:.2f} ms")
    print(f"Status        : {'✓ Solved' if ok else '✗ Failed'}")

    print("\nBenchmarking 50 calls...")
    times = []
    for _ in range(50):
        _, t, _ = mpc.solve(state + np.random.randn(6)*0.01, ref)
        times.append(t)

    print(f"Mean : {np.mean(times):.2f} ms")
    print(f"Std  : {np.std(times):.2f} ms")
    print(f"Min  : {np.min(times):.2f} ms")
    print(f"Max  : {np.max(times):.2f} ms")