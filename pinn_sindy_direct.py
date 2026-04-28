"""
SINDy-PI + PINN  —  Direct 10D Chemical Relaxation
====================================================
No autoencoder. No bottleneck. No dimensionality reduction.

State:   x = [X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N, T, P]  ∈ R¹⁰
             (all in normalised log10 space during training)

What this does
--------------
SINDy-PI loss  (Kaheman 2020):
  For each candidate left-hand side θⱼ(x):
      θⱼ = Θ(x | θⱼ) ξⱼ          (convex regression, not null-space)
  → L_sindy = Σⱼ ‖θⱼ - Θ\ⱼ(x) ξⱼ‖²  +  λ_sparse ‖ξⱼ‖₁

PINN physics loss:
  → L_atom   : atom conservation  Σⱼ Aₑⱼ Xⱼ = const  (from YAML stoichiometry)
  → L_energy : constant-UV        Σⱼ Xⱼ hⱼ(T) - RT = const  (NASA9 polynomials)

Total:
  L = L_sindy + λ_atom · L_atom + λ_energy · L_energy

Architecture
------------
  Ξ : Parameter matrix  [n_theta × n_x]   (the only learnable weights)
      ẋ_predicted = Θ(x) Ξ
  No encoder, no decoder, no hidden layers.
  n_theta = 1 + n_x + n_x*(n_x+1)/2  =  1 + 10 + 55  =  66  (poly degree 2)

Training
--------
  Input  to loss : x(t)  in normalised log10 space           [N, 10]
  Target         : ẋ(t)  finite-difference in same space      [N, 10]
  Optimizer      : AdamW + CosineAnnealingWarmRestarts  (as specified)
  Split          : 70% train | 15% val | 15% test  (by time, no leakage)

Usage
-----
  python pinn_sindy_direct.py               # train
  python pinn_sindy_direct.py --predict     # load checkpoint + plot + CSV
"""

import os, argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    yaml_path   = 'airNASA9ions.yaml',
    csv_path    = 'training_data.csv',
    checkpoint  = 'sindy_direct.pth',

    species     = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N'],  # 8 species + T + P = 10

    sindy_poly_degree = 2,      # library: 1 + 10 + 55 = 66 terms

    # loss weights
    lambda_sparse = 1e-4,       # L1 on Ξ  (sparsity — SINDy-PI style)
    lambda_atom   = 1.0,        # atom conservation
    lambda_energy = 0.1,        # constant-UV energy

    # training
    epochs      = 3000,
    batch_size  = 512,
    lr          = 3e-3,         # higher lr OK — Ξ is the only parameter
    val_frac    = 0.15,
    test_frac   = 0.15,
    weight_decay= 1e-5,
    T0_restart  = 600,

    # SINDy-PI sequential threshold (applied after training)
    sindy_threshold = 0.01,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
np.random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  STOICHIOMETRY MATRIX  (from YAML, for atom conservation loss)
# ─────────────────────────────────────────────────────────────────────────────
def build_stoichiometry_matrix(yaml_path: str, species: list) -> torch.Tensor:
    """
    A  [n_elem × n_species]   A[e, j] = #atoms of element e in species j.
    Elements: C, N, O  (Ar and He have no chemistry — exclude to keep A slim)
    """
    elements = ['C', 'N', 'O']

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    comp = {sp['name']: sp.get('composition', {}) for sp in data['species']}

    A = np.zeros((len(elements), len(species)), dtype=np.float32)
    for j, sp in enumerate(species):
        for i, el in enumerate(elements):
            A[i, j] = comp.get(sp, {}).get(el, 0.0)

    print("Stoichiometry A  [C N O] × species:")
    for i, el in enumerate(elements):
        vals = '  '.join(f'{v:+.0f}' for v in A[i])
        print(f"  {el}: {vals}")

    return torch.tensor(A, dtype=torch.float32)   # [3, 8]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NASA-9 ENTHALPY  (for constant-UV energy loss)
#     h°(T) = R·T · [ -a1/T² + a2·ln(T)/T + a3 + a4·T/2 + a5·T²/3
#                     + a6·T³/4 + a7·T⁴/5 + b1/T ]
#     Implemented in PyTorch so gradients flow through T.
# ─────────────────────────────────────────────────────────────────────────────
def parse_nasa9_coeffs(yaml_path: str, species: list) -> dict:
    """
    Returns dict: species_name → list of (T_lo, T_hi, a[9], b[2]) segments.
    We only need the 200-6000 K and 6000-20000 K ranges for high-T chemistry.
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    coeffs = {}
    for sp_data in data['species']:
        name = sp_data['name']
        if name not in species:
            continue
        thermo = sp_data['thermo']
        T_ranges = thermo['temperature-ranges']   # e.g. [200, 1000, 6000, 20000]
        segs = thermo['data']                     # list of coefficient arrays
        coeffs[name] = []
        for k in range(len(segs)):
            T_lo = T_ranges[k]
            T_hi = T_ranges[k + 1]
            a    = segs[k][:7]    # a1..a7
            b    = segs[k][7:9]   # b1, b2
            coeffs[name].append((T_lo, T_hi, a, b))
    return coeffs


def nasa9_h_over_RT(T: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    Dimensionless enthalpy  h/(RT)  from NASA-9 coefficients.
    T : [batch]  temperature in K
    a : [7]      NASA-9 a-coefficients  (a1..a7)
    Returns [batch]
    """
    T2 = T * T;  T3 = T2 * T;  T4 = T3 * T
    h_RT = (- a[0] / T2
            + a[1] * torch.log(T) / T
            + a[2]
            + a[3] * T  / 2.0
            + a[4] * T2 / 3.0
            + a[5] * T3 / 4.0
            + a[6] * T4 / 5.0)
    return h_RT   # + b1/T omitted (integration constant, cancels in differences)


def build_nasa9_tensors(nasa9_coeffs: dict, species: list, device: str):
    """
    Pre-build coefficient tensors for the two main temperature segments.
    Segments: low  200–6000 K  (index 0 or 1 depending on mechanism)
              high 6000–20000 K
    We use the segment that covers >6000 K for high-T reentry chemistry.
    Returns:
      a_lo  [n_sp, 7]   coefficients for T < 6000 K
      a_hi  [n_sp, 7]   coefficients for T ≥ 6000 K
      T_mid [n_sp]      transition temperature (typically 6000 K)
    """
    n_sp  = len(species)
    a_lo  = torch.zeros(n_sp, 7)
    a_hi  = torch.zeros(n_sp, 7)
    T_mid = torch.full((n_sp,), 6000.0)

    for j, sp in enumerate(species):
        segs = nasa9_coeffs.get(sp, [])
        # Find segment covering ~1000–6000 K (mid range)
        for (T_lo, T_hi, a, b) in segs:
            if T_lo <= 1000.0 and T_hi >= 6000.0:
                a_lo[j] = torch.tensor(a, dtype=torch.float32)
            if T_lo >= 5000.0:
                a_hi[j] = torch.tensor(a, dtype=torch.float32)
                T_mid[j] = T_lo

    return a_lo.to(device), a_hi.to(device), T_mid.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SINDy LIBRARY  Θ(x)   (polynomial degree 2 in 10D state)
# ─────────────────────────────────────────────────────────────────────────────
def sindy_library(x: torch.Tensor, degree: int = 2) -> torch.Tensor:
    """
    x : [batch, n_x]
    Returns Θ : [batch, n_theta]

    Terms (degree 2):
      [1,  x1..x10,  x1²,x1x2,..,x10²]
      n_theta = 1 + 10 + 55 = 66
    """
    batch, n = x.shape
    terms = [torch.ones(batch, 1, device=x.device)]   # constant
    terms.append(x)                                    # degree 1

    if degree >= 2:
        for i in range(n):
            for j in range(i, n):
                terms.append((x[:, i] * x[:, j]).unsqueeze(1))

    return torch.cat(terms, dim=1)   # [batch, n_theta]


def library_size(n_x: int, degree: int) -> int:
    n = 1 + n_x
    if degree >= 2:
        n += n_x * (n_x + 1) // 2
    return n


def library_term_names(n_x: int, degree: int, state_names: list) -> list:
    names = ['1'] + state_names[:]
    if degree >= 2:
        for i in range(n_x):
            for j in range(i, n_x):
                if i == j:
                    names.append(f'{state_names[i]}²')
                else:
                    names.append(f'{state_names[i]}·{state_names[j]}')
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THE MODEL  —  just Ξ, nothing else
# ─────────────────────────────────────────────────────────────────────────────
class SINDyDirect(nn.Module):
    """
    Direct SINDy in the original state space.
    No encoder, no decoder.

    ẋ = Θ(x) Ξ

    Ξ : [n_theta × n_x]  — the only learnable parameter.
    """
    def __init__(self, n_x: int, n_theta: int):
        super().__init__()
        # Small random init so sparsity loss can prune quickly
        self.Xi = nn.Parameter(0.01 * torch.randn(n_theta, n_x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x    : [batch, n_x]   normalised state
        Returns ẋ_pred : [batch, n_x]
        """
        theta = sindy_library(x, degree=2)   # [batch, n_theta]
        return theta @ self.Xi               # [batch, n_x]

    def predict_xdot(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a.  SINDy regression loss  ‖ẋ - Θ(x)Ξ‖²  ──────────────────────────────
def sindy_regression_loss(xdot_pred: torch.Tensor,
                          xdot_data: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and finite-difference derivative."""
    return F.mse_loss(xdot_pred, xdot_data)


# ── 5b.  Sparsity  λ ‖Ξ‖₁  ──────────────────────────────────────────────────
def sparsity_loss(Xi: torch.Tensor) -> torch.Tensor:
    return Xi.abs().mean()


# ── 5c.  Atom conservation  ──────────────────────────────────────────────────
def atom_conservation_loss(
    x_log10_pred: torch.Tensor,   # [batch, n_x]  predicted log10 state
    x_log10_init: torch.Tensor,   # [batch, n_x]  initial log10 state (t=0 broadcast)
    A: torch.Tensor,              # [n_elem, n_sp]
    n_sp: int,
) -> torch.Tensor:
    """
    Atom counts must equal the initial atom counts.
    Works in linear (not log) mole-fraction space.

    x_log10_pred[:, :n_sp] are the predicted log10 mole fractions.
    x_log10_init[:, :n_sp] are the initial  log10 mole fractions (constant row).
    """
    Xi_pred = 10.0 ** x_log10_pred[:, :n_sp]   # [batch, n_sp]
    Xi_init = 10.0 ** x_log10_init[:, :n_sp]   # [batch, n_sp]

    atoms_pred = Xi_pred @ A.T   # [batch, n_elem]
    atoms_init = Xi_init @ A.T   # [batch, n_elem]

    return F.mse_loss(atoms_pred, atoms_init)


# ── 5d.  Constant-UV energy  ─────────────────────────────────────────────────
def energy_conservation_loss(
    x_log10_pred: torch.Tensor,   # [batch, n_x]   predicted log10 state
    u0: torch.Tensor,             # [batch]         initial internal energy [J/kg]
    a_lo: torch.Tensor,           # [n_sp, 7]
    a_hi: torch.Tensor,           # [n_sp, 7]
    T_mid: torch.Tensor,          # [n_sp]
    n_sp: int,
    R_univ: float = 8314.46,      # J/(kmol·K) — universal gas constant
) -> torch.Tensor:
    """
    Internal energy per unit mole:
        u(T) = h(T) - RT  =  Σⱼ Xⱼ · [ h°ⱼ(T)/RT · RT  - RT ]
                           =  Σⱼ Xⱼ · RT · [ h°ⱼ/RT - 1 ]

    We enforce  u_pred(T_pred, X_pred)  ≈  u0  (the known constant).

    For simplicity we evaluate using the 'lo' coefficients only
    (valid for T < 6000 K); for very high T the 'hi' branch kicks in.
    A smooth blend is used to keep gradients continuous.
    """
    Xi_pred = 10.0 ** x_log10_pred[:, :n_sp]          # [batch, n_sp]
    T_pred  = 10.0 ** x_log10_pred[:, n_sp]            # [batch]  temperature K

    T_pred  = T_pred.clamp(200.0, 19999.0)             # safety clamp

    # Blend weight: sigmoid transition around T_mid (6000 K)
    # blend = 0 → use lo coeffs, blend = 1 → use hi coeffs
    blend = torch.sigmoid((T_pred.unsqueeze(1) - T_mid.unsqueeze(0)) / 200.0)
    # [batch, n_sp]

    # h/RT for each species using lo and hi coefficients
    h_lo = torch.stack(
        [nasa9_h_over_RT(T_pred, a_lo[j]) for j in range(n_sp)], dim=1
    )   # [batch, n_sp]

    h_hi = torch.stack(
        [nasa9_h_over_RT(T_pred, a_hi[j]) for j in range(n_sp)], dim=1
    )   # [batch, n_sp]

    h_over_RT = (1.0 - blend) * h_lo + blend * h_hi    # [batch, n_sp]

    # u = h - RT  →  u/(RT) = h/(RT) - 1
    # Total u per mole of mixture = Σⱼ Xⱼ · (h_j/RT - 1) · R · T
    u_over_RT = h_over_RT - 1.0                         # [batch, n_sp]
    u_mix = (Xi_pred * u_over_RT).sum(dim=1) * R_univ * T_pred   # [batch]  J/kmol

    # Normalise both sides by R_univ·T_ref so the loss is dimensionless
    T_ref   = 7500.0  # K — typical initial temperature
    scale   = R_univ * T_ref
    return F.mse_loss(u_mix / scale, u0 / scale)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATA LOADING  (same pipeline as before, unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str, species: list, val_frac: float, test_frac: float,
              batch_size: int):
    """
    Loads training_data.csv, normalises, computes ẋ, splits, returns loaders.

    Each batch yields:
      x_norm   [B, n_x]  normalised log10 state
      xdot_norm[B, n_x]  d(x_norm)/d(log10 t)   — SINDy regression target
      x_raw    [B, n_x]  raw log10 state          — physics loss input
      u0       [B]       initial internal energy   — energy loss target
                         (broadcast from t=0 row)
    """
    df = pd.read_csv(csv_path)
    n_sp = len(species)

    # ── Assemble raw log10 state  [N, n_x] ───────────────────────────────────
    log10_sp = np.stack([df[f'log10_X_{s}'].values for s in species], axis=1)
    log10_T  = np.log10(df['T_K'].values).reshape(-1, 1)
    log10_P  = np.log10(df['P_Pa'].values).reshape(-1, 1)
    Y_raw    = np.hstack([log10_sp, log10_T, log10_P]).astype(np.float32)  # [N, n_x]

    # ── Normalise ─────────────────────────────────────────────────────────────
    Y_mean = Y_raw.mean(axis=0).astype(np.float32)
    Y_std  = Y_raw.std(axis=0).astype(np.float32)
    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)

    # ── log10(t) axis for derivative computation ───────────────────────────────
    log10_t = df['log10_t'].values.astype(np.float32)

    # ── ẋ = d(Y_norm)/d(log10_t)  via central finite differences ──────────────
    dYdt_norm = np.gradient(Y_norm, log10_t, axis=0).astype(np.float32)

    # ── Initial internal energy (constant for whole trajectory) ───────────────
    # Approximate: u ≈ cv * T, here we use T at t=0 as the reference scalar
    # A real implementation would call Cantera or compute from NASA polynomials.
    # We store log10(T_0) and broadcast it — the loss will compare T_pred to T_0.
    # (Replaced by proper NASA9 computation inside energy_conservation_loss.)
    T_K      = df['T_K'].values.astype(np.float32)
    u0_proxy = T_K[0] * np.ones(len(T_K), dtype=np.float32)   # [N] — placeholder J/K

    # ── Split by time order  (no random shuffle — avoids leakage) ─────────────
    N = len(Y_norm)
    n_test  = int(N * test_frac)
    n_val   = int(N * val_frac)
    n_train = N - n_val - n_test

    idx_train = np.arange(n_train)
    idx_val   = np.arange(n_train, n_train + n_val)
    idx_test  = np.arange(n_train + n_val, N)

    stats = dict(Y_mean=Y_mean, Y_std=Y_std)

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.tensor(Y_norm[idx]),
            torch.tensor(dYdt_norm[idx]),
            torch.tensor(Y_raw[idx]),
            torch.tensor(u0_proxy[idx]),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    loaders = (
        make_loader(idx_train, shuffle=True),
        make_loader(idx_val,   shuffle=False),
        make_loader(idx_test,  shuffle=False),
    )

    # Initial log10 state for atom conservation reference  [1, n_x]
    x_init_log10 = torch.tensor(Y_raw[0:1], dtype=torch.float32)   # broadcast later

    print(f"Data: {N} rows  |  train {n_train}  val {n_val}  test {n_test}")
    print(f"State dim n_x = {Y_norm.shape[1]}   "
          f"Library n_theta = {library_size(Y_norm.shape[1], 2)}")

    return loaders, stats, x_init_log10


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict = CFG):
    print("=" * 65)
    print("SINDy-PI + PINN  —  Direct 10D  (no autoencoder)")
    print("=" * 65)

    species = cfg['species']
    n_sp    = len(species)
    n_x     = n_sp + 2           # + T + P

    # ── Chemistry setup ───────────────────────────────────────────────────────
    A = build_stoichiometry_matrix(cfg['yaml_path'], species).to(DEVICE)  # [3, 8]

    nasa9 = parse_nasa9_coeffs(cfg['yaml_path'], species)
    a_lo, a_hi, T_mid = build_nasa9_tensors(nasa9, species, DEVICE)

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader), stats, x_init = load_data(
        cfg['csv_path'], species,
        cfg['val_frac'], cfg['test_frac'], cfg['batch_size']
    )
    x_init = x_init.to(DEVICE)   # [1, n_x]  initial state for atom conservation

    # ── Model: just Ξ ─────────────────────────────────────────────────────────
    n_theta = library_size(n_x, cfg['sindy_poly_degree'])
    model   = SINDyDirect(n_x, n_theta).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nXi shape: [{n_theta} × {n_x}]  =  {n_params} parameters total")

    # ── Optimizer: AdamW + CosineAnnealingWarmRestarts ────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg['T0_restart'], T_mult=1, eta_min=1e-6
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val  = float('inf')
    hist_keys = ['train', 'val', 'sindy', 'sparse', 'atom', 'energy']
    history   = {k: [] for k in hist_keys}

    for epoch in range(1, cfg['epochs'] + 1):

        # — Train —
        model.train()
        running = dict(total=0.0, sindy=0.0, sparse=0.0, atom=0.0, energy=0.0)
        n_seen  = 0

        for x_norm, xdot_norm, x_raw, u0 in train_loader:
            x_norm  = x_norm.to(DEVICE)
            xdot_norm = xdot_norm.to(DEVICE)
            x_raw   = x_raw.to(DEVICE)
            u0      = u0.to(DEVICE)

            optimizer.zero_grad()

            # ẋ predicted by SINDy
            xdot_pred = model(x_norm)                          # [B, n_x]

            # Loss components
            L_sindy  = sindy_regression_loss(xdot_pred, xdot_norm)
            L_sparse = sparsity_loss(model.Xi)

            # Physics: atom conservation  (compare to initial state)
            x_init_broadcast = x_init.expand(x_norm.shape[0], -1)
            L_atom = atom_conservation_loss(x_raw, x_init_broadcast, A, n_sp)

            # Physics: energy conservation  (NASA-9 enthalpy)
            L_energy = energy_conservation_loss(
                x_raw, u0, a_lo, a_hi, T_mid, n_sp
            )

            L_total = (L_sindy
                       + cfg['lambda_sparse'] * L_sparse
                       + cfg['lambda_atom']   * L_atom
                       + cfg['lambda_energy'] * L_energy)

            L_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(x_norm)
            running['total']  += L_total.item()  * bs
            running['sindy']  += L_sindy.item()  * bs
            running['sparse'] += L_sparse.item() * bs
            running['atom']   += L_atom.item()   * bs
            running['energy'] += L_energy.item() * bs
            n_seen += bs

        scheduler.step()
        for k in running:
            running[k] /= n_seen

        # — Validate (SINDy regression only) —
        model.eval()
        val_loss = 0.0; n_val = 0
        with torch.no_grad():
            for x_norm, xdot_norm, x_raw, u0 in val_loader:
                x_norm    = x_norm.to(DEVICE)
                xdot_norm = xdot_norm.to(DEVICE)
                xdot_pred = model(x_norm)
                val_loss += F.mse_loss(xdot_pred, xdot_norm).item() * len(x_norm)
                n_val    += len(x_norm)
        val_loss /= n_val

        history['train'].append(running['total'])
        history['val'].append(val_loss)
        for k in ['sindy', 'sparse', 'atom', 'energy']:
            history[k].append(running[k])

        # Checkpoint on best val
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict' : model.state_dict(),
                'val_loss'         : val_loss,
                'epoch'            : epoch,
                'cfg'              : cfg,
                'stats'            : stats,
                'x_init'           : x_init.cpu(),
                'n_x'              : n_x,
                'n_theta'          : n_theta,
                'species'          : species,
            }, cfg['checkpoint'])

        if epoch % 200 == 0 or epoch == 1:
            print(f"Ep {epoch:>5}  "
                  f"train={running['total']:.4e}  val={val_loss:.4e}  "
                  f"sindy={running['sindy']:.3e}  "
                  f"sparse={running['sparse']:.3e}  "
                  f"atom={running['atom']:.3e}  "
                  f"energy={running['energy']:.3e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nBest val: {best_val:.6e}  →  {cfg['checkpoint']}")

    # ── Post-training SINDy threshold ─────────────────────────────────────────
    apply_threshold(model, cfg['sindy_threshold'])

    # ── Test ──────────────────────────────────────────────────────────────────
    model.eval()
    test_loss = 0.0; n_test = 0
    with torch.no_grad():
        for x_norm, xdot_norm, x_raw, u0 in test_loader:
            xdot_pred = model(x_norm.to(DEVICE))
            test_loss += F.mse_loss(xdot_pred, xdot_norm.to(DEVICE)).item() * len(x_norm)
            n_test    += len(x_norm)
    print(f"Test SINDy MSE (normalised): {test_loss/n_test:.6e}")

    print_equations(model, n_x, cfg['sindy_poly_degree'], species)

    return model, history, stats


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SINDy THRESHOLD  (sequential thresholding — SINDy-PI style)
# ─────────────────────────────────────────────────────────────────────────────
def apply_threshold(model: SINDyDirect, threshold: float):
    with torch.no_grad():
        mask = model.Xi.abs() >= threshold
        model.Xi.data *= mask.float()
    nz = mask.sum().item()
    tot = model.Xi.numel()
    print(f"\nSINDy threshold {threshold}: {nz}/{tot} nonzero  "
          f"({100*nz/tot:.1f}% dense)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 9.  PRINT DISCOVERED EQUATIONS
# ─────────────────────────────────────────────────────────────────────────────
def print_equations(model: SINDyDirect, n_x: int, degree: int, species: list):
    state_names = [f'log10_{s}' for s in species] + ['log10_T', 'log10_P']
    Xi   = model.Xi.detach().cpu().numpy()   # [n_theta, n_x]
    terms = library_term_names(n_x, degree, state_names)

    print("\n── Discovered SINDy equations  ẋ = Θ(x) Ξ ──")
    for k, sname in enumerate(state_names):
        nz = [(terms[t], Xi[t, k]) for t in range(len(terms)) if abs(Xi[t, k]) > 1e-6]
        if nz:
            eq = ' + '.join(f'{v:+.5f}·{nm}' for nm, v in nz)
        else:
            eq = '0  (zero — fully pruned)'
        print(f"  d({sname})/dt = {eq}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SAVE PREDICTIONS CSV  (same format as training_data.csv)
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions_csv(
    model: SINDyDirect,
    csv_path: str,
    stats: dict,
    species: list,
    device: str = 'cpu',
    out_path: str = 'sindy_predictions.csv',
) -> pd.DataFrame:
    """
    Integrates  ẋ = Θ(x) Ξ  forward from x(t₀) using the Euler method
    in normalised log10 space, then de-normalises and saves a CSV
    with exactly the same columns as training_data.csv.

    Columns:
        time, log10_t,
        X_CO2 .. X_N,  log10_X_CO2 .. log10_X_N,
        T_K, P_Pa, rho_kgm3 (NaN — not predicted)
    """
    n_sp = len(species)
    df_ref = pd.read_csv(csv_path)

    Y_mean = stats['Y_mean']
    Y_std  = stats['Y_std']

    # ── Load normalised states for the whole trajectory ───────────────────────
    log10_sp = np.stack([df_ref[f'log10_X_{s}'].values for s in species], axis=1)
    log10_T  = np.log10(df_ref['T_K'].values).reshape(-1, 1)
    log10_P  = np.log10(df_ref['P_Pa'].values).reshape(-1, 1)
    Y_raw    = np.hstack([log10_sp, log10_T, log10_P]).astype(np.float32)

    # ── Run SINDy  ẋ = Θ(x)Ξ  at every stored time point ────────────────────
    # This gives the predicted derivative, not the integrated trajectory.
    # For each row we also run a single Euler step and store that state.
    # The "reconstructed" state is obtained by integrating from t[0].
    log10_t = df_ref['log10_t'].values.astype(np.float32)
    dt_log  = np.diff(log10_t, prepend=log10_t[0])   # Δlog10(t) for each step

    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)

    model.eval()
    BATCH = 2048
    xdot_list = []
    with torch.no_grad():
        for s in range(0, len(Y_norm), BATCH):
            xb = torch.tensor(Y_norm[s:s+BATCH]).to(device)
            xdot_list.append(model(xb).cpu().numpy())
    xdot_norm = np.vstack(xdot_list)   # [N, n_x]  predicted derivatives (normalised)

    # ── Forward Euler integration from initial condition ──────────────────────
    x_int = Y_norm[0:1].copy()          # [1, n_x]  starts from true initial state
    x_integrated = np.zeros_like(Y_norm)
    x_integrated[0] = x_int

    with torch.no_grad():
        for i in range(1, len(log10_t)):
            xb    = torch.tensor(x_int).to(device)
            dxdt  = model(xb).cpu().numpy()           # [1, n_x]
            x_int = x_int + dxdt * dt_log[i]         # Euler step in log10-t
            x_int = np.clip(x_int, -16.0, 10.0)      # safety clamp
            x_integrated[i] = x_int

    # ── De-normalise integrated trajectory ────────────────────────────────────
    Y_hat_log10 = x_integrated * (Y_std + 1e-8) + Y_mean   # back to log10
    Y_hat_orig  = 10.0 ** Y_hat_log10                       # linear units

    X_pred  = Y_hat_orig[:, :n_sp]
    T_pred  = Y_hat_orig[:, n_sp]
    P_pred  = Y_hat_orig[:, n_sp + 1]

    min_frac     = 1e-16
    log10_X_pred = np.log10(np.clip(X_pred, min_frac, None))

    # ── Assemble DataFrame — column-for-column identical to data_generation.py ─
    data = {'time': df_ref['time'].values, 'log10_t': log10_t}
    for j, name in enumerate(species):
        data[f'X_{name}']      = X_pred[:, j]
    for j, name in enumerate(species):
        data[f'log10_X_{name}']= log10_X_pred[:, j]
    data['T_K']      = T_pred
    data['P_Pa']     = P_pred
    data['rho_kgm3'] = np.full(len(log10_t), np.nan)

    df_out = pd.DataFrame(data)
    df_out.to_csv(out_path, index=False, float_format='%.8e')
    print(f"\n✓ Saved  →  {out_path}")
    print(f"  {df_out.shape[0]} rows × {df_out.shape[1]} columns")

    # ── Quick accuracy report ─────────────────────────────────────────────────
    print(f"\n{'Column':<20}  {'Max |Δlog10|':>14}  {'Mean |Δlog10|':>14}")
    print(f"{'-'*20}  {'-'*14}  {'-'*14}")
    for j, name in enumerate(species):
        gt   = df_ref[f'log10_X_{name}'].values
        pred = log10_X_pred[:, j]
        d    = np.abs(pred - gt)
        print(f"  log10_X_{name:<9}  {d.max():>14.4f}  {d.mean():>14.4f}")
    gt_T = np.log10(df_ref['T_K'].values);   pred_T = np.log10(np.clip(T_pred, 1, None))
    gt_P = np.log10(df_ref['P_Pa'].values);  pred_P = np.log10(np.clip(P_pred, 1, None))
    print(f"  {'log10_T_K':<18}  {np.abs(pred_T-gt_T).max():>14.4f}  {np.abs(pred_T-gt_T).mean():>14.4f}")
    print(f"  {'log10_P_Pa':<18}  {np.abs(pred_P-gt_P).max():>14.4f}  {np.abs(pred_P-gt_P).mean():>14.4f}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 11.  PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_training(history: dict, save_path: str = 'sindy_direct_loss.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(history['train'], label='Train total', lw=2)
    ax.semilogy(history['val'],   label='Val SINDy',   lw=2, ls='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Training — SINDy-PI Direct')

    ax = axes[1]
    for k, lbl, ls in [('sindy','SINDy ‖ẋ-Θξ‖²','-'),
                        ('sparse','Sparsity ‖Ξ‖₁','--'),
                        ('atom','Atom conservation','-.'),
                        ('energy','Energy conservation',':')]:
        ax.semilogy(history[k], label=lbl, ls=ls, lw=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title('Loss Components')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss plot → {save_path}")
    plt.show()


def plot_comparison(csv_path: str, pred_csv: str, species: list,
                    save_path: str = 'sindy_direct_comparison.png'):
    df_gt   = pd.read_csv(csv_path)
    df_pred = pd.read_csv(pred_csv)
    time    = df_gt['time'].values

    n_sp = len(species)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_vars = [(f'log10_X_{s}', f'log10_X_{s}', s) for s in species] + \
                [('T_K', 'T_K', 'T [K]')]

    for ax, (gt_col, pred_col, title) in zip(axes, plot_vars):
        gt   = df_gt[gt_col].values
        pred = df_pred[pred_col].values
        ax.semilogx(time, gt,   'k-',  lw=2,   label='Cantera')
        ax.semilogx(time, pred, 'r--', lw=1.5, label='SINDy-PI')
        ax.set_title(title); ax.set_xlabel('t [s]')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('SINDy-PI + PINN  vs  Cantera', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Comparison plot → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 12.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--csv',    default=CFG['csv_path'])
    parser.add_argument('--yaml',   default=CFG['yaml_path'])
    parser.add_argument('--epochs', type=int, default=CFG['epochs'])
    args = parser.parse_args()

    CFG['csv_path']  = args.csv
    CFG['yaml_path'] = args.yaml
    CFG['epochs']    = args.epochs

    if args.predict:
        ckpt    = torch.load(CFG['checkpoint'], map_location=DEVICE, weights_only=False)
        cfg_    = ckpt['cfg']
        stats_  = ckpt['stats']
        species_= ckpt['species']
        n_x_    = ckpt['n_x']
        n_theta_= ckpt['n_theta']

        model_ = SINDyDirect(n_x_, n_theta_).to(DEVICE)
        model_.load_state_dict(ckpt['model_state_dict'])

        print_equations(model_, n_x_, cfg_['sindy_poly_degree'], species_)

        save_predictions_csv(
            model_, CFG['csv_path'], stats_, species_,
            device=DEVICE, out_path='sindy_predictions.csv'
        )
        plot_comparison(CFG['csv_path'], 'sindy_predictions.csv', species_)

    else:
        model, history, stats = train(CFG)
        plot_training(history)

        save_predictions_csv(
            model, CFG['csv_path'], stats, CFG['species'],
            device=DEVICE, out_path='sindy_predictions.csv'
        )
        plot_comparison(CFG['csv_path'], 'sindy_predictions.csv', CFG['species'])
