"""
SINDy-Autoencoder PINN for Chemical Relaxation  —  pinn_sindy_Dt.py
====================================================================
Architecture: Champion, Lusch, Kutz, Brunton — PNAS 2019
             φ: encoder  x(t)  →  z(t)   (latent space)
             ψ: decoder  z(t)  →  x̂(t)  (reconstruction)
             SINDy: ż = Θ(z) Ξ           (sparse dynamics in latent space)

INPUT  : initial concentrations + T + P  AND  elapsed time Δt
OUTPUT : species concentrations + T at time t₀ + Δt

Correct prediction flow:
  1. Encode initial state  z₀ = φ(x(t₀))
  2. Integrate ODE in latent space:  ż = Θ(z) Ξ  from 0 to Δt
  3. Decode at any t:  x̂(t) = ψ(z(t))

Loss (exactly as in the PNAS image, PLUS data fidelity term):
  L = ‖x - ψ(z)‖²                          [reconstruction]
    + λ₁ ‖ẋ - (∇_z ψ(z)) Θ(z^T) Ξ ‖²      [SINDy loss in ẋ]
    + λ₂ ‖(∇_x z) ẋ - Θ(z^T) Ξ ‖²         [SINDy loss in ż]
    + λ₃ ‖Ξ‖₁                              [SINDy sparsity]
    + λ₄ ‖x_pred(t) - x_data(t)‖²          [data fidelity — NEW]

Physics constraints (from data_generation.py + airNASA9ions.yaml):
  - Constant UV (internal energy + volume) — IdealGasReactor
  - Atom conservation: C, N, O, Ar, He
  - Built from YAML stoichiometry matrix

Data pipeline:
  - Input  : x(t₀) = [log10_X_sp..., log10_T, log10_P]  (initial state)
             log10(Δt)  =  log10(t) - log10(t₀)  (time step)
  - Output : x(t)  = [log10_X_sp..., log10_T, log10_P]  (future state)
  - Prediction CSV spans log10_t = -12 → +2  (same format as training_data.csv)
"""

import os
import numpy as np
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # --- files ---
    yaml_path   = '/Users/xiaoxizhou/Downloads/adrian_surf/code/airNASA9ions.yaml',
    csv_path    = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data.csv',
    checkpoint  = 'sindy_autoencoder_Dt.pth',

    # --- species tracked in data_generation.py (must match CSV columns) ---
    species     = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N'],

    # --- network sizes ---
    latent_dim  = 8,     # z dimension  (≥ 2, ≤ n_species+2)
    enc_hidden  = [64, 64, 32],
    dec_hidden  = [32, 64, 64],

    # --- SINDy library: poly degree 2 in z ---
    sindy_poly_degree = 2,

    # --- loss weights ---
    lambda1     = 1e-3,   # SINDy loss in ẋ
    lambda2     = 1e-3,   # SINDy loss in ż
    lambda3     = 1e-5,   # L1 sparsity on Ξ
    lambda4     = 1.0,    # Data fidelity: predicted x(t) vs training data  [NEW]

    # --- physics constraint weights ---
    lambda_atom   = 0.1,   # atom conservation residual
    lambda_energy = 0.1,   # constant-UV energy residual

    # --- training ---
    epochs      = 2000,
    batch_size  = 512,
    lr          = 3e-4,
    val_frac    = 0.15,
    test_frac   = 0.15,
    weight_decay= 1e-5,
    T0_restart  = 500,    # CosineAnnealingWarmRestarts period

    # --- SINDy ODE integration (for prediction) ---
    sindy_threshold = 0.05,
    ode_method  = 'RK45',
    ode_rtol    = 1e-6,
    ode_atol    = 1e-8,

    # --- prediction output time grid ---
    pred_log10_t_min = -12.0,
    pred_log10_t_max =   2.0,
    pred_n_points    = 20000,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PARSE YAML → STOICHIOMETRY MATRIX  (for atom conservation)
# ─────────────────────────────────────────────────────────────────────────────
def build_stoichiometry_matrix(yaml_path: str, species_tracked: list) -> torch.Tensor:
    """
    Returns A  [n_elements × n_species_tracked]  where A[e,s] = number of
    atoms of element e in species s.
    """
    elements = ['C', 'N', 'O', 'Ar', 'He']

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    comp_lookup = {}
    for sp in data['species']:
        comp_lookup[sp['name']] = sp.get('composition', {})

    A = np.zeros((len(elements), len(species_tracked)), dtype=np.float32)
    for j, sp_name in enumerate(species_tracked):
        comp = comp_lookup.get(sp_name, {})
        for i, elem in enumerate(elements):
            A[i, j] = comp.get(elem, 0.0)

    print("Stoichiometry matrix A  [elements × species]:")
    for i, e in enumerate(elements):
        row = '  '.join(f'{v:+.0f}' for v in A[i])
        print(f"  {e:4s}: {row}")

    return torch.tensor(A, dtype=torch.float32)   # [n_elem, n_sp]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_and_normalize(csv_path: str, species: list, val_frac: float, test_frac: float):
    """
    Returns:
      loaders : (train_loader, val_loader, test_loader)
      stats   : dict with X_mean, X_std, Y_mean, Y_std  and  Dt_mean, Dt_std
      raw     : dict with X_raw, Y_raw

    Dataset construction (IC + Δt → future state):
      For every consecutive pair (i, j) where j > i:
        - x0 = state at time[i]   (initial condition, normalised)
        - dt = log10(t[j]) - log10(t[i])   (elapsed log10 time)
        - y  = state at time[j]   (target)

    To keep dataset size manageable we use every pair separated by a fixed
    stride (stride = 1 means consecutive pairs only).  For 20 000 time points
    consecutive pairs → 19 999 training samples, which is sufficient.
    """
    df = pd.read_csv(csv_path)
    N  = len(df)

    # ── Build raw state matrix: [N, n_sp+2] in log10 space ───────────────────
    log10_sp = np.stack(
        [df[f'log10_X_{s}'].values for s in species], axis=1
    ).astype(np.float32)                                           # [N, n_sp]
    log10_T  = np.log10(df['T_K'].values).reshape(-1, 1).astype(np.float32)
    log10_P  = np.log10(df['P_Pa'].values).reshape(-1, 1).astype(np.float32)
    S_raw    = np.hstack([log10_sp, log10_T, log10_P])            # [N, n_x]

    log10_t  = df['log10_t'].values.astype(np.float32)            # [N]

    # ── Normalisation stats (over the state matrix) ───────────────────────────
    Y_mean = S_raw.mean(axis=0); Y_std = S_raw.std(axis=0)        # [n_x]
    S_norm = (S_raw - Y_mean) / (Y_std + 1e-8)                    # [N, n_x]

    # ── Time derivative d(S_norm)/d(log10_t)  for SINDy ẋ / ż losses ─────────
    dSdt_norm = np.gradient(S_norm, log10_t, axis=0).astype(np.float32)

    # ── Build IC-pair dataset: (x0, log10_Dt, x_target) ──────────────────────
    # Use stride=1 (consecutive pairs).  x0 = state at t[i], target = state at t[i+1]
    stride = 1
    idx_i = np.arange(0, N - stride, stride)
    idx_j = idx_i + stride

    X0_norm    = S_norm[idx_i]                               # [M, n_x] initial state
    X0_raw     = S_raw[idx_i]                                # [M, n_x] raw log10
    Xtgt_norm  = S_norm[idx_j]                               # [M, n_x] target state
    Xtgt_raw   = S_raw[idx_j]                                # [M, n_x] raw log10
    log10_Dt   = (log10_t[idx_j] - log10_t[idx_i]).reshape(-1, 1)  # [M, 1]
    dSdt_i     = dSdt_norm[idx_i]                            # [M, n_x] deriv at t_i

    # Normalise Δt separately
    Dt_mean = log10_Dt.mean(); Dt_std = log10_Dt.std()
    log10_Dt_norm = (log10_Dt - Dt_mean) / (Dt_std + 1e-8)

    M = len(idx_i)
    n_test  = int(M * test_frac)
    n_val   = int(M * val_frac)
    n_train = M - n_val - n_test

    idx_all   = np.arange(M)
    idx_train = idx_all[:n_train]
    idx_val   = idx_all[n_train:n_train + n_val]
    idx_test  = idx_all[n_train + n_val:]

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(X0_norm[idx]),           # initial state (normalised)
            torch.tensor(log10_Dt_norm[idx]),     # Δt (normalised)
            torch.tensor(Xtgt_norm[idx]),         # target state (normalised)
            torch.tensor(dSdt_i[idx]),            # time derivative at t_i
            torch.tensor(X0_raw[idx]),            # raw log10 at t_i (physics)
            torch.tensor(Xtgt_raw[idx]),          # raw log10 at t_j (data fidelity)
        )
        return DataLoader(ds, batch_size=CFG['batch_size'], shuffle=shuffle,
                          pin_memory=True, num_workers=0)

    loaders = (
        make_loader(idx_train, shuffle=True),
        make_loader(idx_val,   shuffle=False),
        make_loader(idx_test,  shuffle=False),
    )
    stats = dict(
        Y_mean   = Y_mean,   Y_std   = Y_std,
        Dt_mean  = float(Dt_mean), Dt_std = float(Dt_std),
        log10_t  = log10_t,        # keep for prediction grid
        S_raw    = S_raw,          # keep full trajectory (un-normalised)
        S_norm   = S_norm,
    )
    raw = dict(S_raw=S_raw, log10_t=log10_t)

    print(f"Dataset: {N} time points → {M} IC pairs")
    print(f"  train {n_train}  val {n_val}  test {n_test}")
    print(f"  State dim n_x = {S_raw.shape[1]}   Δt range: "
          f"[{log10_Dt.min():.2f}, {log10_Dt.max():.2f}]  (log10 space)")
    return loaders, stats, raw


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SINDy LIBRARY  Θ(z)
# ─────────────────────────────────────────────────────────────────────────────
def sindy_library(z: torch.Tensor, poly_degree: int = 2) -> torch.Tensor:
    """
    Build polynomial library Θ(z) for each sample.
    z: [batch, latent_dim]
    Returns Θ: [batch, n_library_terms]
    Terms: [1, z1, z2, ..., z1², z1z2, z2², ...]
    """
    batch = z.shape[0]
    terms = [torch.ones(batch, 1, device=z.device)]
    terms.append(z)

    if poly_degree >= 2:
        latent_dim = z.shape[1]
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                terms.append((z[:, i] * z[:, j]).unsqueeze(1))

    return torch.cat(terms, dim=1)   # [batch, n_theta]


def sindy_library_size(latent_dim: int, poly_degree: int) -> int:
    n = 1 + latent_dim
    if poly_degree >= 2:
        n += latent_dim * (latent_dim + 1) // 2
    return n


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENCODER / DECODER  (φ and ψ)
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Generic MLP with LayerNorm + SiLU."""
    def __init__(self, dims: list, final_activation=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i+1]))
                layers.append(nn.SiLU())
        if final_activation is not None:
            layers.append(final_activation)
        self.net = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class SINDyAutoencoder(nn.Module):
    """
    SINDy-Autoencoder  (Champion et al. 2019)

    φ (encoder):  x(t₀)     →  z₀       [n_x → latent_dim]
    ψ (decoder):  z(t)       →  x̂(t)    [latent_dim → n_x]
    Ξ           : ż = Θ(z) Ξ             [n_theta × latent_dim]

    Prediction:
      z₀ = φ(x(t₀))
      Integrate  dz/d(log10_t) = Θ(z) Ξ  from log10_t₀  to  log10_t₀+Δlog10t
      x̂(t) = ψ(z(t))
    """
    def __init__(self, n_x: int, latent_dim: int,
                 enc_hidden: list, dec_hidden: list, poly_degree: int = 2):
        super().__init__()
        self.latent_dim  = latent_dim
        self.poly_degree = poly_degree
        n_theta = sindy_library_size(latent_dim, poly_degree)

        # φ: encoder  (n_x → latent_dim)
        enc_dims = [n_x] + enc_hidden + [latent_dim]
        self.encoder = MLP(enc_dims, final_activation=nn.Tanh())

        # ψ: decoder  (latent_dim → n_x)
        dec_dims = [latent_dim] + dec_hidden + [n_x]
        self.decoder = MLP(dec_dims)

        # Ξ: SINDy coefficients  [n_theta, latent_dim]
        self.Xi = nn.Parameter(0.01 * torch.randn(n_theta, latent_dim))

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """φ: x → z"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """ψ: z → x̂"""
        return self.decoder(z)

    def sindy_zdot(self, z: torch.Tensor) -> torch.Tensor:
        """ż = Θ(z) Ξ,  z: [batch, latent_dim]"""
        theta = sindy_library(z, self.poly_degree)   # [batch, n_theta]
        return theta @ self.Xi                        # [batch, latent_dim]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Autoencoder forward: x → z → x̂  (for reconstruction loss)."""
        z  = self.encode(x)
        xh = self.decode(z)
        return z, xh

    # ------------------------------------------------------------------
    def integrate_latent(
        self,
        z0: torch.Tensor,
        log10_Dt_norm: torch.Tensor,
        Dt_mean: float,
        Dt_std:  float,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """
        Euler integration of  dz/dτ = Θ(z) Ξ  from τ=0 to τ=Δτ
        (in normalised Δt space).

        z0          : [batch, latent_dim]  — initial latent state
        log10_Dt_norm: [batch, 1]           — normalised Δlog10(t)
        Returns z(Δt): [batch, latent_dim]

        Uses fixed-step Euler for differentiability during training.
        (At inference, scipy RK45 is used instead — see predict_trajectory.)
        """
        z   = z0
        tau = log10_Dt_norm.squeeze(-1)   # [batch]

        for _ in range(n_steps):
            dt_step = tau / n_steps
            zdot    = self.sindy_zdot(z)                  # [batch, latent_dim]
            z       = z + zdot * dt_step.unsqueeze(-1)    # Euler step

        return z

    # ------------------------------------------------------------------
    def forward_ic(
        self,
        x0_norm: torch.Tensor,
        log10_Dt_norm: torch.Tensor,
        Dt_mean: float,
        Dt_std:  float,
    ) -> torch.Tensor:
        """
        Full IC-based prediction:
          1. z₀ = φ(x₀)
          2. z(Δt) = integrate(z₀, Δt)
          3. x̂(Δt) = ψ(z(Δt))

        Returns x̂_target: [batch, n_x]  in normalised space.
        """
        z0   = self.encode(x0_norm)                                 # [batch, latent_dim]
        z_t  = self.integrate_latent(z0, log10_Dt_norm,
                                     Dt_mean, Dt_std)               # [batch, latent_dim]
        xhat = self.decode(z_t)                                     # [batch, n_x]
        return xhat


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PHYSICS CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────
def atom_conservation_residual(
    x_hat_log10: torch.Tensor,
    x_ref_log10: torch.Tensor,
    A: torch.Tensor,
    n_species: int,
) -> torch.Tensor:
    """
    Atom conservation: Σ_j A[e,j] * Xi_j = const for each element e.
    x_hat_log10: [batch, n_x]  — predicted state in log10 space
    x_ref_log10: [batch, n_x]  — reference state in log10 space
    A:           [n_elem, n_sp]
    """
    Xi_pred = 10.0 ** x_hat_log10[:, :n_species]
    Xi_ref  = 10.0 ** x_ref_log10[:, :n_species]
    atoms_pred = Xi_pred @ A.T
    atoms_ref  = Xi_ref  @ A.T
    return F.mse_loss(atoms_pred, atoms_ref)


def energy_conservation_residual(
    x_hat_log10: torch.Tensor,
    x_ref_log10: torch.Tensor,
    n_species: int,
) -> torch.Tensor:
    """
    Constant-UV proxy: penalise predicted log10(T) deviating from reference.
    """
    log10_T_pred = x_hat_log10[:, n_species]
    log10_T_ref  = x_ref_log10[:, n_species]
    return F.mse_loss(log10_T_pred, log10_T_ref)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SINDy LOSS TERMS  (Jacobian-based, as in PNAS)
# ─────────────────────────────────────────────────────────────────────────────
def sindy_loss_xdot(
    model: SINDyAutoencoder,
    x: torch.Tensor,
    z: torch.Tensor,
    xdot: torch.Tensor,
) -> torch.Tensor:
    """λ₁: ‖ẋ - J_ψ · Θ(z)Ξ‖²"""
    z_var = z.detach().requires_grad_(True)
    xh = model.decode(z_var)
    n_x = xh.shape[1]

    J_rows = []
    for k in range(n_x):
        gk = torch.autograd.grad(
            xh[:, k].sum(), z_var,
            retain_graph=True, create_graph=True
        )[0]
        J_rows.append(gk.unsqueeze(1))
    J_psi = torch.cat(J_rows, dim=1)                   # [batch, n_x, latent_dim]

    zdot_sindy = model.sindy_zdot(z).unsqueeze(-1)     # [batch, latent_dim, 1]
    xdot_pred  = (J_psi @ zdot_sindy).squeeze(-1)      # [batch, n_x]
    return F.mse_loss(xdot_pred, xdot)


def sindy_loss_zdot(
    model: SINDyAutoencoder,
    x: torch.Tensor,
    z: torch.Tensor,
    xdot: torch.Tensor,
) -> torch.Tensor:
    """λ₂: ‖J_φ · ẋ - Θ(z)Ξ‖²"""
    x_var = x.detach().requires_grad_(True)
    z_var = model.encode(x_var)
    latent_dim = z_var.shape[1]

    J_rows = []
    for k in range(latent_dim):
        gk = torch.autograd.grad(
            z_var[:, k].sum(), x_var,
            retain_graph=True, create_graph=True
        )[0]
        J_rows.append(gk.unsqueeze(1))
    J_phi = torch.cat(J_rows, dim=1)                   # [batch, latent_dim, n_x]

    zdot_data  = (J_phi @ xdot.unsqueeze(-1)).squeeze(-1)
    zdot_sindy = model.sindy_zdot(z)
    return F.mse_loss(zdot_sindy, zdot_data)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FULL LOSS FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(
    model: SINDyAutoencoder,
    x0_norm: torch.Tensor,         # initial state (normalised)
    log10_Dt_norm: torch.Tensor,   # Δlog10(t) (normalised)
    xtgt_norm: torch.Tensor,       # target state (normalised)  — ground truth
    dxdt_norm: torch.Tensor,       # dx/d(log10t) at t₀ (normalised)
    x0_raw: torch.Tensor,          # raw log10 at t₀   (for physics)
    xtgt_raw: torch.Tensor,        # raw log10 at t_j   (for data fidelity)
    A: torch.Tensor,
    n_species: int,
    cfg: dict,
    Dt_mean: float,
    Dt_std:  float,
    compute_sindy: bool = True,
) -> dict:
    """
    Compute all loss terms.

    The autoencoder is trained in two complementary ways:
      (a) Reconstruction  x → z → x̂  (classic AE loss, using x0)
      (b) IC-prediction   x0, Δt → x̂(t)  vs  x_target  [data fidelity — λ₄]
    """
    # ── (a) Reconstruction at t₀: x₀ → z₀ → x̂₀ ─────────────────────────────
    z0, xh0 = model(x0_norm)
    L_recon = F.mse_loss(xh0, x0_norm)

    # ── (b) IC-based prediction: x̂(t) vs training data ─────────────────────
    #    z₀ already computed above; integrate forward
    z_t   = model.integrate_latent(z0, log10_Dt_norm, Dt_mean, Dt_std)
    xhat_t = model.decode(z_t)
    L_data = F.mse_loss(xhat_t, xtgt_norm)     # ← DATA FIDELITY LOSS (λ₄)

    # ── SINDy losses (Jacobian) ───────────────────────────────────────────────
    if compute_sindy:
        L_xdot = sindy_loss_xdot(model, x0_norm, z0, dxdt_norm)
        L_zdot = sindy_loss_zdot(model, x0_norm, z0, dxdt_norm)
    else:
        L_xdot = torch.tensor(0.0, device=x0_norm.device)
        L_zdot = torch.tensor(0.0, device=x0_norm.device)

    # ── SINDy L1 sparsity ────────────────────────────────────────────────────
    L_sparse = model.Xi.abs().mean()

    # ── Physics: atom conservation on predicted future state ─────────────────
    #    De-normalise xhat_t back to log10 space
    Y_mean_t = torch.tensor(cfg.get('Y_mean', np.zeros(x0_norm.shape[1])),
                             device=x0_norm.device, dtype=torch.float32)
    Y_std_t  = torch.tensor(cfg.get('Y_std',  np.ones(x0_norm.shape[1])),
                             device=x0_norm.device, dtype=torch.float32)
    xhat_raw = xhat_t * (Y_std_t + 1e-8) + Y_mean_t     # log10 space prediction

    L_atom   = atom_conservation_residual(xhat_raw, xtgt_raw, A, n_species)
    L_energy = energy_conservation_residual(xhat_raw, xtgt_raw, n_species)

    # ── Total ─────────────────────────────────────────────────────────────────
    L_total = (
        L_recon
        + cfg['lambda4']      * L_data
        + cfg['lambda1']      * L_xdot
        + cfg['lambda2']      * L_zdot
        + cfg['lambda3']      * L_sparse
        + cfg['lambda_atom']  * L_atom
        + cfg['lambda_energy']* L_energy
    )

    return dict(
        total   = L_total,
        recon   = L_recon,
        data    = L_data,
        xdot    = L_xdot,
        zdot    = L_zdot,
        sparse  = L_sparse,
        atom    = L_atom,
        energy  = L_energy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SINDY THRESHOLDING
# ─────────────────────────────────────────────────────────────────────────────
def apply_sindy_threshold(model: SINDyAutoencoder, threshold: float):
    with torch.no_grad():
        mask = model.Xi.abs() >= threshold
        model.Xi.data *= mask.float()
    n_nonzero = mask.sum().item()
    n_total   = model.Xi.numel()
    print(f"SINDy threshold {threshold}: {n_nonzero}/{n_total} nonzero "
          f"({100*n_nonzero/n_total:.1f}% dense)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict = CFG):
    print("=" * 70)
    print("SINDy-Autoencoder PINN — Chemical Relaxation  (IC+Δt → future state)")
    print("=" * 70)

    A = build_stoichiometry_matrix(cfg['yaml_path'], cfg['species']).to(DEVICE)
    n_species = len(cfg['species'])
    n_x = n_species + 2   # species + T + P

    (train_loader, val_loader, test_loader), stats, raw = load_and_normalize(
        cfg['csv_path'], cfg['species'], cfg['val_frac'], cfg['test_frac']
    )

    # Store normalisation stats in cfg so compute_loss can use them
    cfg['Y_mean'] = stats['Y_mean'].tolist()
    cfg['Y_std']  = stats['Y_std'].tolist()
    Dt_mean = stats['Dt_mean']
    Dt_std  = stats['Dt_std']

    model = SINDyAutoencoder(
        n_x        = n_x,
        latent_dim = cfg['latent_dim'],
        enc_hidden = cfg['enc_hidden'],
        dec_hidden = cfg['dec_hidden'],
        poly_degree= cfg['sindy_poly_degree'],
    ).to(DEVICE)

    n_theta     = sindy_library_size(cfg['latent_dim'], cfg['sindy_poly_degree'])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: n_x={n_x}  latent_dim={cfg['latent_dim']}  "
          f"n_theta={n_theta}  params={total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg['T0_restart'], T_mult=1, eta_min=1e-6
    )

    best_val = float('inf')
    history  = {k: [] for k in
                ['train_total','val_total','recon','data','xdot','zdot','sparse']}
    sindy_warmup_epochs = 50

    for epoch in range(1, cfg['epochs'] + 1):
        compute_sindy = (epoch > sindy_warmup_epochs)

        # — Train —
        model.train()
        running = {k: 0.0 for k in ['total','recon','data','xdot','zdot','sparse']}
        n_train = 0

        for x0_b, dt_b, xtgt_b, dxdt_b, x0raw_b, xtgtraw_b in train_loader:
            x0_b      = x0_b.to(DEVICE)
            dt_b      = dt_b.to(DEVICE)
            xtgt_b    = xtgt_b.to(DEVICE)
            dxdt_b    = dxdt_b.to(DEVICE)
            x0raw_b   = x0raw_b.to(DEVICE)
            xtgtraw_b = xtgtraw_b.to(DEVICE)

            optimizer.zero_grad()
            losses = compute_loss(
                model, x0_b, dt_b, xtgt_b, dxdt_b, x0raw_b, xtgtraw_b,
                A, n_species, cfg, Dt_mean, Dt_std, compute_sindy
            )
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(x0_b)
            for k in running:
                running[k] += losses[k].item() * bs
            n_train += bs

        scheduler.step()
        for k in running:
            running[k] /= n_train

        # — Validate (reconstruction only for speed) —
        model.eval()
        val_total = 0.0; n_val = 0
        with torch.no_grad():
            for x0_b, dt_b, xtgt_b, dxdt_b, x0raw_b, xtgtraw_b in val_loader:
                x0_b   = x0_b.to(DEVICE)
                dt_b   = dt_b.to(DEVICE)
                xtgt_b = xtgt_b.to(DEVICE)
                z0, xh0 = model(x0_b)
                z_t     = model.integrate_latent(z0, dt_b, Dt_mean, Dt_std)
                xhat_t  = model.decode(z_t)
                val_total += (F.mse_loss(xh0, x0_b).item() +
                              F.mse_loss(xhat_t, xtgt_b).item()) * len(x0_b)
                n_val += len(x0_b)
        val_total /= n_val

        history['train_total'].append(running['total'])
        history['val_total'].append(val_total)
        for k in ['recon','data','xdot','zdot','sparse']:
            history[k].append(running[k])

        if val_total < best_val:
            best_val = val_total
            torch.save({
                'epoch'            : epoch,
                'model_state_dict' : model.state_dict(),
                'val_loss'         : val_total,
                'cfg'              : cfg,
                'stats'            : stats,
                'species'          : cfg['species'],
                'n_x'              : n_x,
                'n_theta'          : n_theta,
            }, cfg['checkpoint'])

        if epoch % 100 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:>5}/{cfg['epochs']}  "
                  f"train={running['total']:.5f}  val={val_total:.5f}  "
                  f"recon={running['recon']:.5f}  data={running['data']:.5f}  "
                  f"xdot={running['xdot']:.2e}  zdot={running['zdot']:.2e}  "
                  f"sparse={running['sparse']:.2e}  lr={lr_now:.2e}")

    print(f"\nBest val loss: {best_val:.6f}  → {cfg['checkpoint']}")
    apply_sindy_threshold(model, cfg['sindy_threshold'])

    model.eval()
    test_loss = 0.0; n_test = 0
    with torch.no_grad():
        for x0_b, dt_b, xtgt_b, dxdt_b, x0raw_b, xtgtraw_b in test_loader:
            x0_b   = x0_b.to(DEVICE)
            dt_b   = dt_b.to(DEVICE)
            xtgt_b = xtgt_b.to(DEVICE)
            z0, xh0 = model(x0_b)
            z_t     = model.integrate_latent(z0, dt_b, Dt_mean, Dt_std)
            xhat_t  = model.decode(z_t)
            test_loss += (F.mse_loss(xh0, x0_b).item() +
                          F.mse_loss(xhat_t, xtgt_b).item()) * len(x0_b)
            n_test += len(x0_b)
    print(f"Test MSE (recon + data, normalised log10): {test_loss/n_test:.6f}")

    print_sindy_coefficients(model, cfg['latent_dim'], cfg['sindy_poly_degree'])
    return model, history, stats


# ─────────────────────────────────────────────────────────────────────────────
# 10.  PRINT SINDY EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
def print_sindy_coefficients(model: SINDyAutoencoder,
                              latent_dim: int, poly_degree: int):
    Xi = model.Xi.detach().cpu().numpy()
    terms = ['1']
    for i in range(latent_dim):
        terms.append(f'z{i+1}')
    if poly_degree >= 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                terms.append(f'z{i+1}²' if i == j else f'z{i+1}z{j+1}')

    print("\n── Discovered SINDy Equations ż = Θ(z)Ξ ──")
    for k in range(latent_dim):
        nonzero = [(terms[t], Xi[t, k]) for t in range(len(terms))
                   if abs(Xi[t, k]) > 1e-6]
        eq = (' + '.join(f'{v:+.4f}·{nm}' for nm, v in nonzero)
              if nonzero else '0  (fully sparse)')
        print(f"  ż{k+1} = {eq}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  INFERENCE  —  predict_trajectory
# ─────────────────────────────────────────────────────────────────────────────
def predict_trajectory(
    model: SINDyAutoencoder,
    x0_log10: np.ndarray,       # [n_x] — initial state in log10 space
    log10_t_grid: np.ndarray,   # [M]   — output time grid (log10 seconds)
    stats: dict,
    device: str = 'cpu',
) -> np.ndarray:
    """
    Correct SINDy-Autoencoder prediction:
      1. Encode x(t₀) → z₀
      2. Integrate  dz/d(log10_t) = Θ(z) Ξ  from t₀ to every t in grid
      3. Decode z(t) → x̂(t) for each output time

    x0_log10     : initial state [log10(X_CO2), ..., log10(T), log10(P)]
    log10_t_grid : output time axis in log10 seconds (e.g. -12 to +2)
    stats        : normalisation statistics from training

    Returns Y_pred: [M, n_x]  —  log10 values of predicted state
    """
    Y_mean = stats['Y_mean']
    Y_std  = stats['Y_std']

    # Normalise initial condition
    x0_norm = (x0_log10 - Y_mean) / (Y_std + 1e-8)
    x0_t    = torch.tensor(x0_norm, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        z0 = model.encode(x0_t).squeeze(0).cpu().numpy()   # [latent_dim]

    # Get Xi as numpy for scipy integration
    Xi_np = model.Xi.detach().cpu().numpy()   # [n_theta, latent_dim]
    latent_dim  = model.latent_dim
    poly_degree = model.poly_degree

    def sindy_ode(tau, z):
        """ż = Θ(z) Ξ  in numpy (called by scipy solve_ivp)"""
        z_t = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
        theta = sindy_library(z_t, poly_degree).squeeze(0).numpy()  # [n_theta]
        return theta @ Xi_np   # [latent_dim]

    # Reference time: first value on the grid is t₀
    tau0   = log10_t_grid[0]
    tau_span = (tau0, log10_t_grid[-1])
    t_eval   = log10_t_grid

    sol = solve_ivp(
        sindy_ode, tau_span, z0,
        method = CFG['ode_method'],
        t_eval = t_eval,
        rtol   = CFG['ode_rtol'],
        atol   = CFG['ode_atol'],
    )
    # sol.y: [latent_dim, M]

    Z_traj = sol.y.T   # [M, latent_dim]
    Z_t    = torch.tensor(Z_traj, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        X_norm_pred = model.decode(Z_t).cpu().numpy()   # [M, n_x]

    # De-normalise → log10 space
    X_log10_pred = X_norm_pred * (Y_std + 1e-8) + Y_mean   # [M, n_x]
    return X_log10_pred


# ─────────────────────────────────────────────────────────────────────────────
# 12.  SAVE PREDICTIONS CSV  (same format as training_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions_csv(
    model: SINDyAutoencoder,
    csv_path: str,
    stats: dict,
    species: list,
    device: str = 'cpu',
    out_path: str = 'sindy_predictions.csv',
    log10_t_min: float = -12.0,
    log10_t_max: float =   2.0,
    n_points: int = 20000,
) -> pd.DataFrame:
    """
    Generate predictions over a dense log10_t grid from the initial condition
    found at the first row of the training CSV.

    Output columns are identical to training_data.csv (same names, same order,
    float_format='%.8e'):
        time, log10_t,
        X_CO2 ... X_N,
        log10_X_CO2 ... log10_X_N,
        T_K, P_Pa, rho_kgm3  (NaN — not predicted by autoencoder)
    """
    n_species = len(species)
    df_ref    = pd.read_csv(csv_path)

    # ── Initial condition from training data (first row) ─────────────────────
    log10_X0 = np.array([df_ref[f'log10_X_{s}'].values[0] for s in species],
                         dtype=np.float32)
    log10_T0 = np.log10(df_ref['T_K'].values[0])
    log10_P0 = np.log10(df_ref['P_Pa'].values[0])
    x0_log10 = np.hstack([log10_X0, [log10_T0, log10_P0]])    # [n_x]

    print(f"\nInitial condition (t = 10^{log10_t_min:.1f} s):")
    print(f"  log10_T = {log10_T0:.3f}  →  T = {10**log10_T0:.1f} K")
    print(f"  log10_P = {log10_P0:.3f}  →  P = {10**log10_P0:.1f} Pa")
    for j, s in enumerate(species):
        print(f"  log10_X_{s:4s} = {log10_X0[j]:.3f}")

    # ── Build dense output time grid ─────────────────────────────────────────
    log10_t_grid = np.linspace(log10_t_min, log10_t_max, n_points).astype(np.float32)
    time_grid    = 10.0 ** log10_t_grid

    # ── Run prediction via SINDy ODE integration ──────────────────────────────
    print(f"\nIntegrating SINDy ODE from log10_t={log10_t_min} to {log10_t_max} "
          f"({n_points} points)...")
    X_log10_pred = predict_trajectory(
        model, x0_log10, log10_t_grid, stats, device=device
    )   # [M, n_x]

    # ── Unpack columns ────────────────────────────────────────────────────────
    log10_X_pred = X_log10_pred[:, :n_species]           # [M, n_sp]
    log10_T_pred = X_log10_pred[:, n_species]            # [M]
    log10_P_pred = X_log10_pred[:, n_species + 1]        # [M]

    # Linear values (clip to avoid 10^-inf)
    min_frac = 1e-16
    X_pred = np.clip(10.0 ** log10_X_pred, min_frac, None)
    T_pred = 10.0 ** log10_T_pred
    P_pred = 10.0 ** log10_P_pred

    # ── Assemble DataFrame — column order mirrors data_generation.py exactly ──
    data = {'time': time_grid, 'log10_t': log10_t_grid}

    for j, name in enumerate(species):
        data[f'X_{name}'] = X_pred[:, j]

    for j, name in enumerate(species):
        data[f'log10_X_{name}'] = log10_X_pred[:, j]

    data['T_K']      = T_pred
    data['P_Pa']     = P_pred
    data['rho_kgm3'] = np.full(n_points, np.nan)

    df_out = pd.DataFrame(data)
    df_out.to_csv(out_path, index=False, float_format='%.8e')

    print(f"✓ Saved predictions → {out_path}")
    print(f"  Shape : {df_out.shape[0]} rows × {df_out.shape[1]} columns")
    print(f"  Columns: {list(df_out.columns)}")

    # ── Reconstruction quality (compare against training CSV at overlapping times)
    _print_reconstruction_quality(df_out, df_ref, species, n_species)

    return df_out


def _print_reconstruction_quality(df_pred, df_ref, species, n_species):
    """Compare predicted log10 values against ground truth at training times."""
    print("\n── Reconstruction quality (vs Cantera ground truth) ──")
    print(f"  {'Column':<18}  {'Max |Δlog10|':>14}  {'Mean |Δlog10|':>14}")
    print(f"  {'-'*18}  {'-'*14}  {'-'*14}")

    # Interpolate predictions to reference times
    log10_t_pred = df_pred['log10_t'].values
    log10_t_ref  = df_ref['log10_t'].values

    for j, name in enumerate(species):
        pred_interp = np.interp(log10_t_ref,
                                log10_t_pred, df_pred[f'log10_X_{name}'].values)
        gt   = df_ref[f'log10_X_{name}'].values
        diff = np.abs(pred_interp - gt)
        print(f"  log10_X_{name:<9}  {diff.max():>14.4f}  {diff.mean():>14.4f}")

    pred_T_interp = np.interp(log10_t_ref, log10_t_pred, np.log10(df_pred['T_K'].values))
    gt_T   = np.log10(df_ref['T_K'].values)
    diff_T = np.abs(pred_T_interp - gt_T)
    print(f"  {'log10_T_K':<18}  {diff_T.max():>14.4f}  {diff_T.mean():>14.4f}")

    pred_P_interp = np.interp(log10_t_ref, log10_t_pred, np.log10(df_pred['P_Pa'].values))
    gt_P   = np.log10(df_ref['P_Pa'].values)
    diff_P = np.abs(pred_P_interp - gt_P)
    print(f"  {'log10_P_Pa':<18}  {diff_P.max():>14.4f}  {diff_P.mean():>14.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_training(history: dict, save_path: str = 'sindy_training_loss.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(history['train_total'], label='Train total', linewidth=2)
    ax.semilogy(history['val_total'],   label='Val total',   linewidth=2, ls='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.set_title('Total Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    styles = [
        ('recon',  'Reconstruction ‖x-ψ(z)‖²',        '-'),
        ('data',   'Data fidelity ‖x̂(t)-x_data‖² (λ₄)', '-'),
        ('xdot',   'SINDy ẋ (λ₁)',                     '--'),
        ('zdot',   'SINDy ż (λ₂)',                     '-.'),
        ('sparse', 'Sparsity ‖Ξ‖₁ (λ₃)',               ':'),
    ]
    for k, lbl, ls in styles:
        vals = [v for v in history.get(k, []) if v > 0]
        if vals:
            ax.semilogy(history[k], label=lbl, linestyle=ls, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.set_title('Loss Components')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved loss plot → {save_path}")
    plt.show()


def plot_reconstruction(
    model: SINDyAutoencoder,
    csv_path: str,
    stats: dict,
    species: list,
    device: str = 'cpu',
    save_path: str = 'sindy_reconstruction.png',
    log10_t_min: float = -12.0,
    log10_t_max: float =   2.0,
    n_points: int = 5000,
):
    """Plot ground truth vs SINDy ODE prediction on dense time grid."""
    df_ref    = pd.read_csv(csv_path)
    n_species = len(species)

    # Build initial condition
    log10_X0 = np.array([df_ref[f'log10_X_{s}'].values[0] for s in species],
                         dtype=np.float32)
    log10_T0 = np.log10(df_ref['T_K'].values[0])
    log10_P0 = np.log10(df_ref['P_Pa'].values[0])
    x0_log10 = np.hstack([log10_X0, [log10_T0, log10_P0]])

    log10_t_grid = np.linspace(log10_t_min, log10_t_max, n_points)
    time_grid    = 10.0 ** log10_t_grid

    X_log10_pred = predict_trajectory(model, x0_log10, log10_t_grid, stats, device)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    plot_cols = species[:8] + ['T']

    for idx, name in enumerate(plot_cols):
        ax = axes[idx]
        if name in species:
            j = species.index(name)
            gt_log10 = df_ref[f'log10_X_{name}'].values
            gt_t     = df_ref['log10_t'].values
            ax.plot(gt_t,        gt_log10,               'k-',  lw=2, label='Cantera')
            ax.plot(log10_t_grid, X_log10_pred[:, j],   'r--', lw=1.5, label='SINDy')
            ax.set_ylabel('log10(X)')
        else:  # Temperature
            gt_log10 = np.log10(df_ref['T_K'].values)
            gt_t     = df_ref['log10_t'].values
            ax.plot(gt_t,        gt_log10,                        'k-',  lw=2, label='Cantera')
            ax.plot(log10_t_grid, X_log10_pred[:, n_species],    'r--', lw=1.5, label='SINDy')
            ax.set_ylabel('log10(T/K)')
        ax.set_title(name)
        ax.set_xlabel('log10(t/s)')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('SINDy-Autoencoder (IC+Δt) vs Cantera', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved reconstruction plot → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 14.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true',
                        help='Load checkpoint and predict (skip training)')
    parser.add_argument('--csv',    default=CFG['csv_path'])
    parser.add_argument('--yaml',   default=CFG['yaml_path'])
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    args = parser.parse_args()

    CFG['csv_path']  = args.csv
    CFG['yaml_path'] = args.yaml
    CFG['epochs']    = args.epochs

    if args.predict:
        print("Loading checkpoint for prediction...")
        ckpt     = torch.load(CFG['checkpoint'], map_location=DEVICE, weights_only=False)
        cfg_     = ckpt['cfg']
        stats_   = ckpt['stats']
        species_ = ckpt['species']
        n_x_     = ckpt['n_x']

        model_ = SINDyAutoencoder(
            n_x        = n_x_,
            latent_dim = cfg_['latent_dim'],
            enc_hidden = cfg_['enc_hidden'],
            dec_hidden = cfg_['dec_hidden'],
            poly_degree= cfg_['sindy_poly_degree'],
        ).to(DEVICE)
        model_.load_state_dict(ckpt['model_state_dict'])

        print_sindy_coefficients(model_, cfg_['latent_dim'], cfg_['sindy_poly_degree'])
        plot_reconstruction(model_, CFG['csv_path'], stats_, species_,
                            device=DEVICE,
                            log10_t_min=CFG['pred_log10_t_min'],
                            log10_t_max=CFG['pred_log10_t_max'],
                            n_points=5000)
        save_predictions_csv(
            model      = model_,
            csv_path   = CFG['csv_path'],
            stats      = stats_,
            species    = species_,
            device     = DEVICE,
            out_path   = 'sindy_predictions.csv',
            log10_t_min= CFG['pred_log10_t_min'],
            log10_t_max= CFG['pred_log10_t_max'],
            n_points   = CFG['pred_n_points'],
        )

    else:
        model, history, stats = train(CFG)
        plot_training(history)
        plot_reconstruction(model, CFG['csv_path'], stats, CFG['species'],
                            device=DEVICE,
                            log10_t_min=CFG['pred_log10_t_min'],
                            log10_t_max=CFG['pred_log10_t_max'],
                            n_points=5000)
        save_predictions_csv(
            model      = model,
            csv_path   = CFG['csv_path'],
            stats      = stats,
            species    = CFG['species'],
            device     = DEVICE,
            out_path   = 'sindy_predictions.csv',
            log10_t_min= CFG['pred_log10_t_min'],
            log10_t_max= CFG['pred_log10_t_max'],
            n_points   = CFG['pred_n_points'],
        )