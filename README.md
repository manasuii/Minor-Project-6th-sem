# Surrogate-MPC for Quadrotor Attitude Stabilisation

A high-performance, physics-informed benchmarking framework comparing Surrogate Model Predictive Control (Surrogate-MPC) paradigms for 3-DOF quadrotor attitude stabilisation. This codebase delivers a comprehensive, **three-way approximator comparison** between a Residual Neural Network (**ResNN**), Gaussian Processes (**GP**), and Polynomial Chaos Expansion (**PCE**).

By learning the residual dynamics of a high-fidelity nonlinear plant, these data-driven surrogates enable microsecond-level model evaluations, making real-time, non-linear MPC computationally feasible on resource-constrained embedded hardware.

---

## 🚀 Key Features & Architectural Enhancements

* **Three-Way Approximator Comparison:** Seamless integration and evaluation of **ResNN** (SiLU MLP), **GP** (GPyTorch / multi-output independent GPs), and **PCE** (Orthogonal Polynomial Chaos via `chaospy`).
* **Physics-Informed Residual Learning:** All surrogates target the *residual error* between a simplified nominal model and the true plant dynamics, maximizing data efficiency and ensuring physical grounding.
* **True Closed-Loop Baseline MPC:** Features a fully realized nonlinear `ODEMPCController` (using RK4 integration inside an IPOPT loop) alongside classical PID and Linear MPC for rigorous benchmark testing.
* **Unified Interface (`surrogate_common.py`):** Single source of truth for the dataset, preprocessing, evaluation logic, and unified model wrappers ensuring a perfectly level playing field.
* **Embedded-Ready Execution:** Focuses strictly on a 6-state attitude model ($[\phi, \theta, \psi, p, q, r]^T$) optimizing for exact-Jacobian computational graphs within CasADi to smash the 50 ms real-time control budget.

---

## 📊 Pipeline Architecture & Objectives

The codebase maps directly to three primary research objectives:
1.  **O1: Simulation & Dataset Generation:** Validation of the plant via conservation laws and generation of representative trajectories.
2.  **O2: Surrogate Learning & Open-Loop Verification:** Training, sample-efficiency analysis, and latency benchmarking of the three approximators.
3.  **O3: Closed-Loop Control & Benchmarking:** Deployment of the surrogates inside an SQP/OSQP MPC loop compared against classical and optimal baselines across 6 deployment scenarios.

---

## Individual Stages
#,Command,Primary Output/Artifacts,Objective
1,python validate_simulator.py,Energy/momentum conservation report & ODE timing logs,O1
2,python generate_dataset.py,"data/train|val|test.npz, normstats.npz",O1
3,python train_nn_surrogate.py,"models/nn_surrogate.pth (+ cfg), training/parity plots",O2
4,python train_gp_surrogate.py,"models/gp_surrogate_0..5.pth, gp_cfg.npz",O2
5,python train_pce_surrogate.py,models/pce_surrogate.pkl,O2
6,python evaluate_surrogates.py,"plots/rollout_error.png, sample_efficiency.png, latency_comparison.png, results/surrogate_comparison.csv",O2
7,python mpc_surrogate.py,Builds and benchmarks the Surrogate-MPC controllers,O3
8,python baseline_controllers.py,"Evaluates PID, Lin-MPC, and Full ODE-MPC benchmarks",O3
9,python run_experiments.py,"Scenario plots (plots/exp1..6_*.png), results/summary_table.csv, results/realtime_feasibility.csv",O3

## 🛠️ Installation & Setup

This repository is completely open-source and ready for CPU-only or GPU-accelerated environments.

## Repository Structure
├── data/                    # Generated training, validation, and test datasets
├── models/                  # Saved weights and configurations (.pth, .pkl, .npz)
├── plots/                   # Multi-step rollouts, sample-efficiency, and trajectory plots
├── results/                 # Comparative CSVs and real-time feasibility metrics
├── surrogate_common.py      # Core shared wrappers, normalization logic, and target definition
├── quadrotor_simulator.py   # High-fidelity nonlinear plant dynamics
├── run_all.py               # Master orchestration script
└── requirements.txt         # Project dependencies

1. Clone the repository:
   
   ```bash
   git clone [https://github.com/your-username/surrogate-mpc-quadrotor.git](https://github.com/your-username/surrogate-mpc-quadrotor.git)
   cd surrogate-mpc-quadrotor
