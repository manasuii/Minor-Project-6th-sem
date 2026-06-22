# Surrogate-MPC for Quadrotor Attitude Stabilisation — Corrected Codebase

This codebase fulfils the three objectives of the proposal, with the headline
**three-way approximator comparison (ResNN vs GP vs PCE)** now actually delivered.

---

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

(Everything is open-source; CPU is sufficient. A GPU is used automatically if present.)

---

## 2. Run order (chronological)

Run from the project root. Either run the whole pipeline:

```bash
python run_all.py
```

…or run each stage individually, in this order:

| # | Command | Produces | Objective |
|---|---------|----------|-----------|
| 1 | `python validate_simulator.py`   | energy/momentum conservation + ODE timing | O1 |
| 2 | `python generate_dataset.py`     | `data/train|val|test.npz`, `normstats.npz` | O1 |
| 3 | `python train_nn_surrogate.py`   | `models/nn_surrogate.pth` (+cfg), training/parity plots | O2 |
| 4 | `python train_gp_surrogate.py`   | `models/gp_surrogate_0..5.pth`, `gp_cfg.npz` | O2 |
| 5 | `python train_pce_surrogate.py`  | `models/pce_surrogate.pkl` | O2 |
| 6 | `python evaluate_surrogates.py`  | `plots/rollout_error.png`, `sample_efficiency.png`, `latency_comparison.png`, `results/surrogate_comparison.csv` | O2 |
| 7 | `python mpc_surrogate.py`        | builds + benchmarks Surrogate-MPC | O3 |
| 8 | `python baseline_controllers.py` | sanity-checks PID / Lin-MPC / ODE-MPC | O3 |
| 9 | `python run_experiments.py`      | `plots/exp1..6_*.png`, `results/summary_table.csv`, `results/realtime_feasibility.csv` | O3 |

Steps 3–5 must finish before step 6; steps 2–5 must finish before steps 7–9.

> **Runtime note.** `evaluate_surrogates.py` retrains every model on
> {1k, 5k, 10k, 25k} subsets for the sample-efficiency curve. Set
> `RUN_SAMPLE_EFFICIENCY = False` at the top of that file to skip it for a
> quick pass.

---

## 3. What changed and why

**New files**
- `surrogate_common.py` — single source of truth for the dataset, the shared
  *physics-informed residual* learning target, the model classes, and uniform
  `NNSurrogate / GPSurrogate / PCESurrogate` wrappers so all three are compared
  apples-to-apples.
- `train_pce_surrogate.py` — the **Polynomial Chaos Expansion** approximator
  (order-4, chaospy) that was missing entirely.
- `validate_simulator.py` — the proposal's "validate via energy + timing" step
  (torque-free energy/|H| conservation + RK45 per-call timing).

**Rewritten files**
- `evaluate_surrogates.py` — now a genuine **three-way** open-loop study:
  one-step RMSE table, multi-step rollout vs true ODE, the **sample-efficiency
  curve** (previously only promised in the docstring), and latency — all for
  ResNN, GP **and** PCE, with a saved comparison CSV.
- `train_gp_surrogate.py` — GP now learns the **same residual target** as the
  NN (so the comparison is fair), uses relative paths, saves a reloadable
  format, and exposes a reusable trainer.
- `train_nn_surrogate.py` — fixed the hardcoded `D:\MINOR\...` paths (now
  relative), aligned the docstring with the actual `[32,32]` SiLU residual MLP,
  and exposed a reusable trainer for the sweep.
- `baseline_controllers.py` — removed the fake "ODE-MPC timing estimate" and
  added a **real closed-loop `ODEMPCController`** (true RK4-integrated nonlinear
  dynamics inside an IPOPT loop = proposal baseline B3). All horizons set to
  N = 5 to match the proposal.
- `run_experiments.py` — **six** scenarios (added an isolated yaw-axis step),
  **all four** controllers on every scenario, the **mismatch test now runs
  every controller** (not surrogate-only), plus a real-time-feasibility table.
- `run_all.py` — pipeline order updated to include validation + PCE.

**Minimal edits**
- `mpc_surrogate.py` — one import line repointed to `surrogate_common`
  (validated SQP+OSQP solver otherwise untouched).
- `quadrotor_simulator.py`, `generate_dataset.py` — unchanged.

---

## 4. Note on dimensionality and the NN architecture

- The state `x = [phi, theta, psi, p, q, r]` is a **6-state attitude model
  (3 rotational DOF)**. Translational dynamics are out of scope (attitude
  stabilisation). `validate_simulator.py` documents this explicitly.
- The proposal scoped a "4-layer tanh MLP"; the validated implementation uses a
  compact **`[32,32]` SiLU residual MLP**, which embeds into CasADi as a small
  exact-Jacobian graph and keeps the MPC solve time well under the 50 ms budget.
  The residual-learning principle is preserved exactly. If you prefer to match
  the proposal text verbatim, change `hidden_dims` in
  `surrogate_common.QuadrotorNNSurrogate` and retrain.
