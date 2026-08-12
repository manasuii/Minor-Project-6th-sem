#surrogate_common.py
import os
import pickle

import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

from quadrotor_simulator import delta_linear, simulate_trajectory, PARAMS  # noqa: F401

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
MODEL_DIR = "models"
N_INDUCING_DEFAULT = 500

CHANNEL_NAMES = ["phi", "theta", "psi", "p", "q", "r"]
RESIDUAL_NAMES = ["d_phi", "d_theta", "d_psi", "d_p", "d_q", "d_r"]


# ════════════════════════════════════════════════════════════════════════
#  Data + common target
# ════════════════════════════════════════════════════════════════════════
def nonlinear_residual(X, Y):
    """Common learning target: full one-step delta minus linear physics base."""
    return np.asarray(Y, dtype=np.float64) - delta_linear(X)


def load_splits(data_dir=DATA_DIR):
    """Load train/val/test plus normalisation stats. Returns a dict."""
    train = np.load(os.path.join(data_dir, "train.npz"))
    val   = np.load(os.path.join(data_dir, "val.npz"))
    test  = np.load(os.path.join(data_dir, "test.npz"))
    return {
        "X_train": train["X"], "Y_train": train["Y"],
        "X_val":   val["X"],   "Y_val":   val["Y"],
        "X_test":  test["X"],  "Y_test":  test["Y"],
    }


def load_subset(n_train, data_dir=DATA_DIR, seed=42):
    """Deterministic random subset of the training split (for sample-efficiency)."""
    train = np.load(os.path.join(data_dir, "train.npz"))
    X, Y = train["X"], train["Y"]
    n_train = min(n_train, len(X))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n_train, replace=False)
    return X[idx], Y[idx]


def input_norm_stats(X):
    """Mean/std of the 9-D input, used to normalise NN/GP inputs."""
    return X.mean(axis=0).astype(np.float32), (X.std(axis=0) + 1e-8).astype(np.float32)


def residual_norm_stats(X, Y):
    """Mean/std of the nonlinear residual target (NN/GP output normalisation)."""
    R = nonlinear_residual(X, Y)
    return R.mean(axis=0).astype(np.float32), (R.std(axis=0) + 1e-8).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
#  Model class definitions
# ════════════════════════════════════════════════════════════════════════
class QuadrotorNNSurrogate(torch.nn.Module):
    """
    Residual feed-forward surrogate.

    Architecture : Linear(9->32) -> SiLU -> Linear(32->32) -> SiLU -> Linear(32->6)
    Predicts the NORMALISED nonlinear residual given the normalised (state, input).
    """

    def __init__(self, in_dim=9, out_dim=6, hidden_dims=(32, 32)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [torch.nn.Linear(prev, h), torch.nn.SiLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class SVGPModel(ApproximateGP):
    """Sparse Variational GP (one per output channel) with RBF + Matern kernel."""

    def __init__(self, inducing_points):
        var_dist = CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ════════════════════════════════════════════════════════════════════════
#  PCE helpers (chaospy)
# ════════════════════════════════════════════════════════════════════════
def pce_normalise_inputs(X, lower, upper):
    """Map physical inputs into [-1, 1]^9 (well-conditioned PCE domain)."""
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    return 2.0 * (np.asarray(X, dtype=np.float64) - lower) / (upper - lower) - 1.0


# ════════════════════════════════════════════════════════════════════════
#  Uniform surrogate wrappers
# ════════════════════════════════════════════════════════════════════════
class _SurrogateBase:
    """Common interface: predict_residual -> predict_delta -> rollout."""

    def predict_residual(self, X):
        raise NotImplementedError

    def predict_delta(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return delta_linear(X) + self.predict_residual(X)

    def rollout(self, x0, u_sequence):
        u_sequence = np.atleast_2d(u_sequence)
        T = len(u_sequence)
        states = np.zeros((T + 1, 6))
        states[0] = x0
        for t in range(T):
            xu = np.concatenate([states[t], u_sequence[t]]).reshape(1, -1)
            states[t + 1] = states[t] + self.predict_delta(xu)[0]
        return states


class NNSurrogate(_SurrogateBase):
    name = "ResNN"

    def __init__(self, model_path=None, cfg_path=None, model=None, cfg=None):
        model_path = model_path or os.path.join(MODEL_DIR, "nn_surrogate.pth")
        cfg_path   = cfg_path   or os.path.join(MODEL_DIR, "nn_surrogate_cfg.npz")
        if model is None:
            model = QuadrotorNNSurrogate().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model = model.to(DEVICE).eval()
        cfg = cfg if cfg is not None else np.load(cfg_path)
        self.X_mean = cfg["X_mean"].astype(np.float32)
        self.X_std  = cfg["X_std"].astype(np.float32)
        self.Y_mean = cfg["Y_mean"].astype(np.float32)
        self.Y_std  = cfg["Y_std"].astype(np.float32)

    def predict_residual(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        Xn = (X - self.X_mean) / self.X_std
        with torch.no_grad():
            out = self.model(torch.tensor(Xn).float().to(DEVICE)).cpu().numpy()
        return out * self.Y_std + self.Y_mean


class GPSurrogate(_SurrogateBase):
    name = "GP"

    def __init__(self, model_dir=MODEL_DIR, models=None, likelihoods=None, cfg=None):
        cfg = cfg if cfg is not None else np.load(os.path.join(model_dir, "gp_cfg.npz"))
        self.X_mean = cfg["X_mean"].astype(np.float32)
        self.X_std  = cfg["X_std"].astype(np.float32)
        self.Y_mean = cfg["Y_mean"].astype(np.float32)
        self.Y_std  = cfg["Y_std"].astype(np.float32)
        n_ind = int(cfg["n_inducing"]) if "n_inducing" in cfg else N_INDUCING_DEFAULT

        if models is None:
            models, likelihoods = [], []
            for ch in range(6):
                state = torch.load(os.path.join(model_dir, f"gp_surrogate_{ch}.pth"),
                                   map_location=DEVICE)
                dummy = torch.zeros(n_ind, 9)
                m = SVGPModel(dummy).to(DEVICE)
                m.load_state_dict(state["model"])
                lk = gpytorch.likelihoods.GaussianLikelihood().to(DEVICE)
                lk.load_state_dict(state["likelihood"])
                models.append(m.eval())
                likelihoods.append(lk.eval())
        self.models = models
        self.likelihoods = likelihoods

    def predict_residual(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        Xn = torch.tensor((X - self.X_mean) / self.X_std).float().to(DEVICE)
        preds = np.zeros((len(X), 6), dtype=np.float64)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for ch in range(6):
                mean = self.likelihoods[ch](self.models[ch](Xn)).mean.cpu().numpy()
                preds[:, ch] = mean * self.Y_std[ch] + self.Y_mean[ch]
        return preds


class PCESurrogate(_SurrogateBase):
    name = "PCE"

    def __init__(self, model_path=None, approx=None, lower=None, upper=None):
        model_path = model_path or os.path.join(MODEL_DIR, "pce_surrogate.pkl")
        if approx is None:
            with open(model_path, "rb") as fh:
                blob = pickle.load(fh)
            approx = blob["approx"]
            lower, upper = blob["lower"], blob["upper"]
        self.approx = approx
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

    def predict_residual(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xn = pce_normalise_inputs(X, self.lower, self.upper)
        out = np.asarray(self.approx(*Xn.T))      # chaospy -> shape (6, N)
        if out.ndim == 1:
            out = out.reshape(-1, 1)
        return out.T                              # (N, 6)


# ════════════════════════════════════════════════════════════════════════
#  Convenience loaders
# ════════════════════════════════════════════════════════════════════════
def load_all_available(model_dir=MODEL_DIR, verbose=True):
    """Load whichever of {ResNN, GP, PCE} have been trained. Returns ordered dict."""
    from collections import OrderedDict
    out = OrderedDict()
    attempts = [
        ("ResNN", lambda: NNSurrogate()),
        ("GP",    lambda: GPSurrogate(model_dir)),
        ("PCE",   lambda: PCESurrogate()),
    ]
    for name, ctor in attempts:
        try:
            out[name] = ctor()
            if verbose:
                print(f"  loaded surrogate: {name}")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"  (skipping {name}: {exc})")
    return out
