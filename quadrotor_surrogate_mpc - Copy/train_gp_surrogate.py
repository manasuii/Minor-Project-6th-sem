"""
train_gp_surrogate.py  (physics-informed version)
==================================================
GP trains on:  delta_residual = delta_x - delta_linear
at prediction: delta_x = delta_linear + GP_prediction

This matches exactly what the NN does after the physics-informed fix,
making the NN vs GP comparison apples-to-apples.
"""

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import (CholeskyVariationalDistribution,
                                   VariationalStrategy)
from torch.utils.data import TensorDataset, DataLoader
import os, time
from quadrotor_simulator import PARAMS

N_GP_TRAIN  = 10_000
N_INDUCING  = 500
EPOCHS_GP   = 100
BATCH_SIZE  = 512
LR_GP       = 0.01
SEED        = 42

torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Physical parameters ──────────────────────────────────────
Ixx = PARAMS["Ixx"];  Iyy = PARAMS["Iyy"];  Izz = PARAMS["Izz"]
DT  = PARAMS["dt"]


def linear_delta(X_raw):
    """
    Compute the hover-linearised one-step delta for a batch of samples.
    X_raw : (N, 9)  — [phi,theta,psi,p,q,r, tau_phi,tau_theta,tau_psi]
    Returns: (N, 6) — linearised delta_x
    """
    phi   = X_raw[:, 0];  theta = X_raw[:, 1];  psi = X_raw[:, 2]
    p     = X_raw[:, 3];  q     = X_raw[:, 4];  r   = X_raw[:, 5]
    tphi  = X_raw[:, 6];  ttheta= X_raw[:, 7];  tpsi= X_raw[:, 8]

    # Linearised angular accelerations (cross-product terms dropped at hover)
    p_dot = tphi   / Ixx
    q_dot = ttheta / Iyy
    r_dot = tpsi   / Izz

    d_phi   = p   * DT
    d_theta = q   * DT
    d_psi   = r   * DT
    d_p     = p_dot * DT
    d_q     = q_dot * DT
    d_r     = r_dot * DT

    return np.stack([d_phi, d_theta, d_psi, d_p, d_q, d_r], axis=1)


# ── SVGP Model ───────────────────────────────────────────────
class SVGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        vd = CholeskyVariationalDistribution(inducing_points.size(0))
        vs = VariationalStrategy(self, inducing_points, vd,
                                  learn_inducing_locations=True)
        super().__init__(vs)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() +
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ── Data loading ─────────────────────────────────────────────
def load_gp_data(n_train=N_GP_TRAIN):
    train = np.load("data/train.npz")
    test  = np.load("data/test.npz")
    stats = np.load("data/normstats.npz")

    X_mean = stats["X_mean"].astype(np.float32)
    X_std  = stats["X_std"].astype(np.float32)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train["X"]), size=n_train, replace=False)

    X_tr_raw = train["X"][idx].astype(np.float64)
    Y_tr_raw = train["Y"][idx].astype(np.float64)   # full residual delta_x

    X_te_raw = test["X"].astype(np.float64)
    Y_te_raw = test["Y"].astype(np.float64)

    # ── Physics-informed target: subtract linear prediction ──
    lin_tr   = linear_delta(X_tr_raw)
    lin_te   = linear_delta(X_te_raw)

    Y_tr_nl  = (Y_tr_raw - lin_tr).astype(np.float32)   # nonlinear residual only
    Y_te_nl  = (Y_te_raw - lin_te).astype(np.float32)

    # Normalise inputs only (targets kept in physical units for GP)
    Xtr_norm = ((X_tr_raw - X_mean) / X_std).astype(np.float32)
    Xte_norm = ((X_te_raw - X_mean) / X_std).astype(np.float32)

    return (torch.tensor(Xtr_norm), torch.tensor(Y_tr_nl),
            torch.tensor(Xte_norm), Y_te_raw, lin_te,
            X_mean, X_std)


# ── Train one GP per output channel ─────────────────────────
def train_one_gp(ch, Xtr, Ytr, epochs=EPOCHS_GP):
    y_ch = Ytr[:, ch]
    ind_idx = torch.randperm(len(Xtr))[:N_INDUCING]
    ind_pts = Xtr[ind_idx].clone().to(DEVICE)

    model      = SVGPModel(ind_pts).to(DEVICE)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)

    model.train(); likelihood.train()
    opt = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": likelihood.parameters()},
    ], lr=LR_GP)
    mll    = gpytorch.mlls.VariationalELBO(likelihood, model,
                                            num_data=len(Xtr))
    loader = DataLoader(TensorDataset(Xtr.to(DEVICE), y_ch.to(DEVICE)),
                        batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            loss = -mll(model(xb), yb)
            loss.backward()
            opt.step()
        if epoch % 25 == 0:
            print(f"    ch={ch}  epoch={epoch:3d}  ELBO={-loss.item():.4f}")

    return model, likelihood


# ── Evaluate ─────────────────────────────────────────────────
def evaluate_gp(models, likelihoods, Xte_norm, Y_te_raw, lin_te):
    names = ["Δphi","Δtheta","Δpsi","Δp","Δq","Δr"]
    print("\n── GP Test RMSE (physical units, with linear baseline added back) ──")
    all_pred = []

    for i, (m, lh) in enumerate(zip(models, likelihoods)):
        m.eval(); lh.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            dist      = lh(m(Xte_norm.to(DEVICE)))
            nl_pred   = dist.mean.cpu().numpy()        # nonlinear residual
            nl_std    = dist.stddev.cpu().numpy()

        # Add linear prediction back → full delta_x prediction
        full_pred = nl_pred + lin_te[:, i]
        rmse = np.sqrt(np.mean((full_pred - Y_te_raw[:, i])**2))
        print(f"  {names[i]:8s}: RMSE = {rmse:.2e}  "
              f"mean_uncertainty = {nl_std.mean():.2e}")
        all_pred.append(full_pred)

    return np.stack(all_pred, axis=1)


# ── Save prediction wrapper ───────────────────────────────────
def build_gp_predictor(models, likelihoods, X_mean, X_std, lin_delta_fn):
    """
    Returns a callable: state(6,), u(3,) → delta_x(6,)
    Matches the interface expected by run_experiments.py
    """
    def predict(state, u):
        xu = np.concatenate([state, u]).astype(np.float32)
        xu_norm = torch.tensor((xu - X_mean) / X_std).unsqueeze(0).to(DEVICE)

        # Linear baseline
        lin = lin_delta_fn(xu.reshape(1, 9).astype(np.float64))[0]

        # GP nonlinear residual
        nl_pred = np.zeros(6)
        for i, (m, lh) in enumerate(zip(models, likelihoods)):
            m.eval(); lh.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                nl_pred[i] = lh(m(xu_norm)).mean.item()

        return lin + nl_pred

    return predict


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Physics-Informed GP Surrogate Training")
    print(f"Training on nonlinear residual only (delta_x - delta_linear)")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    (Xtr, Ytr_nl, Xte_norm, Y_te_raw,
     lin_te, X_mean, X_std) = load_gp_data()
    print(f"GP training samples : {len(Xtr):,}")
    print(f"Nonlinear residual  : mean abs = "
          f"{np.abs(Ytr_nl.numpy()).mean():.2e}  "
          f"(vs full delta mean = "
          f"{np.abs(Y_te_raw).mean():.2e})")

    models, likelihoods = [], []
    t0 = time.time()

    for ch in range(6):
        print(f"\nTraining GP channel {ch} ...")
        m, lh = train_one_gp(ch, Xtr, Ytr_nl)
        torch.save({"model": m.state_dict(),
                    "likelihood": lh.state_dict()},
                   f"models/gp_pi_surrogate_{ch}.pth")
        models.append(m)
        likelihoods.append(lh)

    print(f"\nTotal training time : {time.time()-t0:.1f} s")

    np.savez("models/gp_pi_cfg.npz",
             X_mean=X_mean, X_std=X_std, n_inducing=N_INDUCING)

    evaluate_gp(models, likelihoods, Xte_norm, Y_te_raw, lin_te)

    print("\n✓ Physics-informed GP training complete")
    print("  Saved: models/gp_pi_surrogate_0.pth ... _5.pth")