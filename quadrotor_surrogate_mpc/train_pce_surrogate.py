import os
import pickle
import time

import numpy as np
import chaospy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_dataset import BOUNDS
from surrogate_common import (
    DATA_DIR, MODEL_DIR, load_splits, load_subset, nonlinear_residual,
    pce_normalise_inputs, RESIDUAL_NAMES, ZERO_TOL,
)

# ─────────────────────────────────────────
#  Hyperparameters
# ─────────────────────────────────────────
N_PCE_TRAIN = 10_000
PCE_ORDER   = 4                  # headline model (unchanged — keeps run_all.py happy)
SWEEP_ORDERS = (1, 2, 3, 4, 5)   # accuracy-latency frontier
RANDOM_SEED = 42
RUN_SWEEP   = True

LOWER = np.array([v[0] for v in BOUNDS.values()], dtype=np.float64)
UPPER = np.array([v[1] for v in BOUNDS.values()], dtype=np.float64)


# ─────────────────────────────────────────
#  Fit  (unchanged interface)
# ─────────────────────────────────────────
def fit_pce(X, Y, order=PCE_ORDER, lower=LOWER, upper=UPPER, verbose=False):
    """
    Fit a vector-valued PCE mapping normalised input -> nonlinear residual.
    Returns (approx polynomial, info-dict).
    """
    Xn = pce_normalise_inputs(X, lower, upper)             # (N, 9) in [-1,1]
    R  = nonlinear_residual(X, Y)                          # (N, 6)

    dist = chaospy.J(*[chaospy.Uniform(-1, 1) for _ in range(Xn.shape[1])])
    expansion = chaospy.generate_expansion(order, dist)
    if verbose:
        print(f"  PCE order {order}: {len(expansion)} basis terms, "
              f"fitting on {len(Xn):,} samples...")
    if len(expansion) > 0.5 * len(Xn):
        print(f"  ! WARNING: {len(expansion)} terms vs {len(Xn)} samples — "
              f"regression is under-determined, expect overfitting")

    t0 = time.time()
    approx = chaospy.fit_regression(expansion, Xn.T, R)    # least-squares
    info = {"n_terms": len(expansion), "order": order, "fit_time": time.time() - t0}
    return approx, info


# ─────────────────────────────────────────
#  Evaluate  (unchanged interface, plus live-channel aware summary)
# ─────────────────────────────────────────
def _eval(approx, X, lower=LOWER, upper=UPPER):
    Xn = pce_normalise_inputs(X, lower, upper)
    out = np.asarray(approx(*Xn.T))            # (6, N)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out.T                               # (N, 6)


def evaluate_pce(approx, X_test, Y_test, verbose=True):
    pred = _eval(approx, X_test)
    true = nonlinear_residual(X_test, Y_test)
    live = true.std(axis=0) > ZERO_TOL

    if verbose:
        print("\n── PCE Test-set nonlinear-residual RMSE (physical units) ──")
        for i, nm in enumerate(RESIDUAL_NAMES):
            rmse = np.sqrt(np.mean((pred[:, i] - true[:, i]) ** 2))
            tag = "" if live[i] else "   (structurally zero — excluded)"
            print(f"  {nm:9s}: RMSE = {rmse:.2e}{tag}")

    overall_all  = np.sqrt(np.mean((pred - true) ** 2))
    overall_live = np.sqrt(np.mean((pred[:, live] - true[:, live]) ** 2))
    if verbose:
        print(f"\n  Overall (all 6 channels)   : {overall_all:.2e}")
        print(f"  Overall (5 live channels)  : {overall_live:.2e}   <- report this one")
    return pred, true, overall_live


# ─────────────────────────────────────────
#  CasADi export — the HONEST latency measurement
# ─────────────────────────────────────────
def pce_to_casadi(approx, lower=LOWER, upper=UPPER):
    """
    Compile the PCE into a CasADi Function.

    This matters. `approx(*Xn.T)` goes through chaospy's Python/numpoly evaluation,
    which carries enormous interpreter overhead — that is almost certainly where the
    ~19 ms figure came from, NOT from the intrinsic cost of the polynomial. A PCE is
    just a sum of monomials; once compiled into the same CasADi graph the MPC uses
    for every other surrogate, it may well be microseconds. Comparing a chaospy
    Python call against a CasADi-embedded MLP is not a like-for-like benchmark and a
    reviewer will say so.
    """
    import casadi as ca
    exps = np.asarray(approx.exponents)                      # (n_terms, 9)
    coefs = np.asarray(approx.coefficients, dtype=np.float64)  # (n_terms, 6)
    if coefs.ndim == 1:
        coefs = coefs.reshape(-1, 1)

    x = ca.SX.sym("x", 9)
    xn = 2.0 * (x - ca.DM(lower)) / (ca.DM(upper) - ca.DM(lower)) - 1.0

    expr = ca.SX.zeros(coefs.shape[1])
    for e, c in zip(exps, coefs):
        m = ca.SX(1.0)
        for i in range(9):
            if e[i] > 0:
                m = m * xn[i] ** int(e[i])
        expr = expr + ca.DM(c) * m
    return ca.Function("pce_residual", [x], [expr])


def benchmark_python(approx, n_trials=200):
    x = np.zeros((1, 9))
    for _ in range(5):
        _eval(approx, x)
    t = time.perf_counter()
    for _ in range(n_trials):
        _eval(approx, x)
    return (time.perf_counter() - t) / n_trials * 1e6      # microseconds


def benchmark_casadi(fn, n_trials=10_000):
    import casadi as ca
    x = ca.DM.zeros(9)
    for _ in range(100):
        fn(x)
    t = time.perf_counter()
    for _ in range(n_trials):
        fn(x)
    return (time.perf_counter() - t) / n_trials * 1e6, fn.n_nodes()


# ─────────────────────────────────────────
#  Accuracy-latency sweep
# ─────────────────────────────────────────
def sweep(Xsub, Ysub, X_test, Y_test, orders=SWEEP_ORDERS):
    """
    Fit PCE at each order and record (terms, RMSE, python latency, casadi latency).

    This is the experiment that makes the paper's argument honestly. Comparing
    ResNN (1504 mul-adds) against order-4 PCE (715 terms) is not a controlled
    comparison — PCE is being handed a far larger compute budget. Sweeping the
    order recovers the frontier and lets you say where each surrogate class sits
    at a MATCHED budget, rather than conceding that you chose the worse model.
    """
    rows = []
    print("\n" + "=" * 74)
    print("ACCURACY-LATENCY SWEEP")
    print("=" * 74)
    print(f"{'order':>5} {'terms':>7} {'test RMSE':>12} {'fit(s)':>8} "
          f"{'py(us)':>10} {'casadi(us)':>11} {'nodes':>8}")
    print("-" * 74)

    for k in orders:
        approx, info = fit_pce(Xsub, Ysub, order=k, verbose=False)
        _, _, rmse = evaluate_pce(approx, X_test, Y_test, verbose=False)
        py_us = benchmark_python(approx)
        try:
            ca_us, nodes = benchmark_casadi(pce_to_casadi(approx))
        except Exception as exc:               # noqa: BLE001
            ca_us, nodes = float("nan"), -1
            print(f"  (casadi export failed at order {k}: {exc})")

        rows.append({"order": k, "terms": info["n_terms"], "rmse": rmse,
                     "fit_time": info["fit_time"], "py_us": py_us,
                     "casadi_us": ca_us, "nodes": nodes})
        print(f"{k:>5} {info['n_terms']:>7} {rmse:>12.3e} {info['fit_time']:>8.1f} "
              f"{py_us:>10.1f} {ca_us:>11.1f} {nodes:>8}")

        with open(os.path.join(MODEL_DIR, f"pce_surrogate_order{k}.pkl"), "wb") as fh:
            pickle.dump({"approx": approx, "lower": LOWER, "upper": UPPER,
                         "order": k}, fh)

    np.savez(os.path.join(MODEL_DIR, "pce_sweep.npz"),
             **{key: np.array([r[key] for r in rows]) for key in rows[0]})
    return rows


def plot_pareto(rows, others=None):
    """others: list of (name, latency_us, rmse) for ResNN / GP / SparsePoly."""
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    lat = [r["casadi_us"] for r in rows]
    err = [r["rmse"] for r in rows]
    ax.plot(lat, err, "o-", color="darkorange", linewidth=2, markersize=7,
            label="PCE (order 1-5)", zorder=3)
    for r in rows:
        ax.annotate(f"  p={r['order']}\n  {r['terms']} terms",
                    (r["casadi_us"], r["rmse"]), fontsize=7, va="center")

    for name, l, e in (others or []):
        ax.scatter([l], [e], s=90, marker="s", zorder=4, label=name)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Inference latency (us, CasADi-compiled)")
    ax.set_ylabel("Test residual RMSE (live channels)")
    ax.set_title("Surrogate accuracy-latency frontier")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/pce_pareto.png", dpi=150)
    plt.close()
    print("\nSaved: plots/pce_pareto.png")


# ─────────────────────────────────────────
#  Main
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("Training PCE Surrogate for Quadrotor Attitude Dynamics")
    print(f"Order {PCE_ORDER} expansion, {N_PCE_TRAIN:,} training points")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    s = load_splits(DATA_DIR)
    Xsub, Ysub = load_subset(N_PCE_TRAIN, DATA_DIR, seed=RANDOM_SEED)

    # ── headline model: unchanged, same path, same pickle keys ────────────
    t0 = time.time()
    approx, info = fit_pce(Xsub, Ysub, order=PCE_ORDER, verbose=True)
    print(f"PCE fit time: {time.time() - t0:.1f} s  ({info['n_terms']} terms)")

    evaluate_pce(approx, s["X_test"], s["Y_test"])

    with open(os.path.join(MODEL_DIR, "pce_surrogate.pkl"), "wb") as fh:
        pickle.dump({"approx": approx, "lower": LOWER, "upper": UPPER,
                     "order": PCE_ORDER}, fh)
    print("\n✓ PCE Surrogate training complete")
    print("  Saved: models/pce_surrogate.pkl")

    # ── the frontier ──────────────────────────────────────────────────────
    if RUN_SWEEP:
        rows = sweep(Xsub, Ysub, s["X_test"], s["Y_test"])

        others = []
        try:
            from surrogate_common import load_all_available
            for name, sur in load_all_available(verbose=False).items():
                if name == "PCE":
                    continue
                pred = sur.predict_residual(s["X_test"])
                true = nonlinear_residual(s["X_test"], s["Y_test"])
                live = true.std(axis=0) > ZERO_TOL
                e = np.sqrt(np.mean((pred[:, live] - true[:, live]) ** 2))
                x1 = s["X_test"][:1]
                for _ in range(20):
                    sur.predict_residual(x1)
                t = time.perf_counter()
                for _ in range(500):
                    sur.predict_residual(x1)
                l = (time.perf_counter() - t) / 500 * 1e6
                others.append((name, l, e))
                print(f"  {name:12s}  {l:8.1f} us   RMSE {e:.3e}")
        except Exception as exc:               # noqa: BLE001
            print(f"  (could not benchmark other surrogates: {exc})")

        plot_pareto(rows, others)


if __name__ == "__main__":
    main()