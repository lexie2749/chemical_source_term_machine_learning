"""
SINDy-PI + PINN  —  Direct 10D with Arrhenius exp(-c/T) Library
================================================================
Same architecture as pinn_sindy_direct.py, **except** the SINDy
candidate library Θ(x) is augmented with Arrhenius-style basis
functions

        exp( -c_q / T )            for q = 1 ... Q

where T is the *physical* (de-normalised) temperature in K and the
c_q are characteristic activation temperatures Ea/R covering the
range relevant to high-T air / Mars-entry chemistry.

Why?
----
Real elementary reaction rates take the form

        dX_i/dt = Σ_r  ν_ir · A_r · T^n_r · exp(-E_a / RT) · Π_j X_j^{v_jr}

A pure polynomial SINDy library can never reproduce the
exp(-E_a / RT) factor — no finite linear combination of monomials
in T equals an exponential of 1/T.  Adding a handful of exp(-c/T)
basis functions lets Ξ pick out the activation energy(ies) that
actually drive the dynamics.

State, atom-conservation loss, NASA-9 energy loss and the
Euler-integration predictor are identical to pinn_sindy_direct.py.

Library
-------
        Θ(x) = [ 1,
                 x_1 ... x_{n_x},                          (linear)
                 x_1², x_1·x_2, ..., x_{n_x}²,             (quadratic)
                 exp(-c_1/T), ..., exp(-c_Q/T) ]           (Arrhenius)

n_theta  =  1 + n_x + n_x(n_x+1)/2 + Q

Usage
-----
  python pinn_sindy_exp.py               # train
  python pinn_sindy_exp.py --predict     # load checkpoint + plot + CSV
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
    checkpoint  = 'sindy_exp.pth',

    species     = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N'],  # 8 species + T + P = 10

    sindy_poly_degree = 2,      # polynomial part: 1 + 10 + 55 = 66 terms

    # Arrhenius activation temperatures Ea/R (in K) for exp(-c/T) basis terms.
    # Spread to cover the Ea/R values seen in high-T air / Mars-entry chemistry:
    #   ~3 000 K  : low-barrier exchange reactions (e.g. O + N2 → NO + N at high T)
    #   ~30 000 K : NO formation channels
    #   ~60 000 K : O2 and CO2 dissociation
    #   ~100 000 K: N2 dissociation tail
    #   ~150 000 K: CO dissociation
    exp_scales  = [3000.0, 30000.0, 60000.0, 100000.0, 150000.0],

    # loss weights
    lambda_sparse = 1e-4,
    lambda_atom   = 1.0,
    lambda_energy = 0.1,

    # training
    epochs      = 3000,
    batch_size  = 512,
    lr          = 3e-3,
    val_frac    = 0.15,
    test_frac   = 0.15,
    weight_decay= 1e-5,
    T0_restart  = 600,

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
    Elements: C, N, O  (Ar / He have no chemistry — excluded).
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

    return torch.tensor(A, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  NASA-9 ENTHALPY  (for constant-UV energy loss)
# ─────────────────────────────────────────────────────────────────────────────
def parse_nasa9_coeffs(yaml_path: str, species: list) -> dict:
    """species_name → list of (T_lo, T_hi, a[7], b[2]) NASA-9 segments."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    coeffs = {}
    for sp_data in data['species']:
        name = sp_data['name']
        if name not in species:
            continue
        thermo = sp_data['thermo']
        T_ranges = thermo['temperature-ranges']
        segs = thermo['data']
        coeffs[name] = []
        for k in range(len(segs)):
            T_lo = T_ranges[k]
            T_hi = T_ranges[k + 1]
            a    = segs[k][:7]
            b    = segs[k][7:9]
            coeffs[name].append((T_lo, T_hi, a, b))
    return coeffs


def nasa9_h_over_RT(T: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Dimensionless enthalpy  h°(T)/RT  from NASA-9 a-coefficients."""
    T2 = T * T;  T3 = T2 * T;  T4 = T3 * T
    h_RT = (- a[0] / T2
            + a[1] * torch.log(T) / T
            + a[2]
            + a[3] * T  / 2.0
            + a[4] * T2 / 3.0
            + a[5] * T3 / 4.0
            + a[6] * T4 / 5.0)
    return h_RT


def build_nasa9_tensors(nasa9_coeffs: dict, species: list, device: str):
    """Pack coefficients for the 1000–6000 K and ≥5000 K segments."""
    n_sp  = len(species)
    a_lo  = torch.zeros(n_sp, 7)
    a_hi  = torch.zeros(n_sp, 7)
    T_mid = torch.full((n_sp,), 6000.0)

    for j, sp in enumerate(species):
        segs = nasa9_coeffs.get(sp, [])
        for (T_lo, T_hi, a, b) in segs:
            if T_lo <= 1000.0 and T_hi >= 6000.0:
                a_lo[j] = torch.tensor(a, dtype=torch.float32)
            if T_lo >= 5000.0:
                a_hi[j] = torch.tensor(a, dtype=torch.float32)
                T_mid[j] = T_lo

    return a_lo.to(device), a_hi.to(device), T_mid.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SINDy LIBRARY  Θ(x_norm, T_actual)   polynomial(deg 2) + exp(-c/T)
# ─────────────────────────────────────────────────────────────────────────────
def sindy_library(x_norm: torch.Tensor,
                  T_actual: torch.Tensor,
                  degree: int,
                  exp_scales) -> torch.Tensor:
    """
    Build the candidate library Θ.

    x_norm     : [batch, n_x]   normalised log10 state
    T_actual   : [batch]        physical temperature (K), de-normalised
    degree     : polynomial degree (1 or 2)
    exp_scales : iterable of activation temperatures c (K) for exp(-c/T)

    Returns Θ : [batch, n_theta]
        n_theta = 1 + n_x + n_x(n_x+1)/2 + len(exp_scales)
    """
    batch, n = x_norm.shape
    terms = [torch.ones(batch, 1, device=x_norm.device)]   # constant
    terms.append(x_norm)                                   # linear

    if degree >= 2:
        for i in range(n):
            for j in range(i, n):
                terms.append((x_norm[:, i] * x_norm[:, j]).unsqueeze(1))

    # Arrhenius-style exp(-c / T) basis functions
    inv_T = 1.0 / T_actual.clamp(min=100.0)                # [batch]
    for c in exp_scales:
        terms.append(torch.exp(-float(c) * inv_T).unsqueeze(1))

    return torch.cat(terms, dim=1)                         # [batch, n_theta]


def library_size(n_x: int, degree: int, n_exp: int) -> int:
    n = 1 + n_x
    if degree >= 2:
        n += n_x * (n_x + 1) // 2
    n += n_exp
    return n


def library_term_names(n_x: int, degree: int, state_names: list, exp_scales) -> list:
    names = ['1'] + state_names[:]
    if degree >= 2:
        for i in range(n_x):
            for j in range(i, n_x):
                if i == j:
                    names.append(f'{state_names[i]}²')
                else:
                    names.append(f'{state_names[i]}·{state_names[j]}')
    for c in exp_scales:
        names.append(f'exp(-{float(c):.0f}/T)')
    return names


# ─────────────────────────────────────────────────────────────────────────────
# 4.  THE MODEL  —  Ξ  + frozen de-normalisation buffers
# ─────────────────────────────────────────────────────────────────────────────
class SINDyExp(nn.Module):
    """
    ẋ_norm  =  Θ_exp(x_norm, T_actual)  ·  Ξ

    Y_mean / Y_std are registered as buffers so that the model can
    recover T_actual from x_norm during training **and** during the
    stand-alone Euler integration in save_predictions_csv (where only
    the normalised state is propagated).

    Ξ is the only learnable parameter.
    """
    def __init__(self,
                 n_x: int,
                 n_theta: int,
                 n_sp: int,
                 Y_mean: np.ndarray,
                 Y_std:  np.ndarray,
                 exp_scales: list,
                 poly_degree: int = 2):
        super().__init__()
        self.Xi          = nn.Parameter(0.01 * torch.randn(n_theta, n_x))
        self.n_sp        = n_sp
        self.poly_degree = poly_degree
        self.exp_scales  = list(exp_scales)
        self.register_buffer('Y_mean',
                             torch.tensor(np.asarray(Y_mean), dtype=torch.float32))
        self.register_buffer('Y_std',
                             torch.tensor(np.asarray(Y_std),  dtype=torch.float32))

    def _T_actual(self, x_norm: torch.Tensor) -> torch.Tensor:
        idx_T   = self.n_sp                                       # column for log10(T)
        log10_T = x_norm[:, idx_T] * (self.Y_std[idx_T] + 1e-8) + self.Y_mean[idx_T]
        return (10.0 ** log10_T).clamp(min=200.0, max=20000.0)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        T_actual = self._T_actual(x_norm)
        theta    = sindy_library(x_norm, T_actual, self.poly_degree, self.exp_scales)
        return theta @ self.Xi

    def predict_xdot(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.forward(x_norm)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sindy_regression_loss(xdot_pred: torch.Tensor,
                          xdot_data: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(xdot_pred, xdot_data)


def sparsity_loss(Xi: torch.Tensor) -> torch.Tensor:
    return Xi.abs().mean()


def atom_conservation_loss(
    x_log10_pred: torch.Tensor,
    x_log10_init: torch.Tensor,
    A: torch.Tensor,
    n_sp: int,
) -> torch.Tensor:
    Xi_pred = 10.0 ** x_log10_pred[:, :n_sp]
    Xi_init = 10.0 ** x_log10_init[:, :n_sp]
    atoms_pred = Xi_pred @ A.T
    atoms_init = Xi_init @ A.T
    return F.mse_loss(atoms_pred, atoms_init)


def energy_conservation_loss(
    x_log10_pred: torch.Tensor,
    u0: torch.Tensor,
    a_lo: torch.Tensor,
    a_hi: torch.Tensor,
    T_mid: torch.Tensor,
    n_sp: int,
    R_univ: float = 8314.46,
) -> torch.Tensor:
    Xi_pred = 10.0 ** x_log10_pred[:, :n_sp]
    T_pred  = 10.0 ** x_log10_pred[:, n_sp]
    T_pred  = T_pred.clamp(200.0, 19999.0)

    blend = torch.sigmoid((T_pred.unsqueeze(1) - T_mid.unsqueeze(0)) / 200.0)

    h_lo = torch.stack(
        [nasa9_h_over_RT(T_pred, a_lo[j]) for j in range(n_sp)], dim=1
    )
    h_hi = torch.stack(
        [nasa9_h_over_RT(T_pred, a_hi[j]) for j in range(n_sp)], dim=1
    )
    h_over_RT = (1.0 - blend) * h_lo + blend * h_hi

    u_over_RT = h_over_RT - 1.0
    u_mix = (Xi_pred * u_over_RT).sum(dim=1) * R_univ * T_pred

    T_ref = 7500.0
    scale = R_univ * T_ref
    return F.mse_loss(u_mix / scale, u0 / scale)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str, species: list, val_frac: float, test_frac: float,
              batch_size: int):
    df = pd.read_csv(csv_path)
    n_sp = len(species)

    log10_sp = np.stack([df[f'log10_X_{s}'].values for s in species], axis=1)
    log10_T  = np.log10(df['T_K'].values).reshape(-1, 1)
    log10_P  = np.log10(df['P_Pa'].values).reshape(-1, 1)
    Y_raw    = np.hstack([log10_sp, log10_T, log10_P]).astype(np.float32)

    Y_mean = Y_raw.mean(axis=0).astype(np.float32)
    Y_std  = Y_raw.std(axis=0).astype(np.float32)
    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)

    log10_t = df['log10_t'].values.astype(np.float32)
    dYdt_norm = np.gradient(Y_norm, log10_t, axis=0).astype(np.float32)

    T_K      = df['T_K'].values.astype(np.float32)
    u0_proxy = T_K[0] * np.ones(len(T_K), dtype=np.float32)

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

    x_init_log10 = torch.tensor(Y_raw[0:1], dtype=torch.float32)

    print(f"Data: {N} rows  |  train {n_train}  val {n_val}  test {n_test}")
    return loaders, stats, x_init_log10


# ─────────────────────────────────────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict = CFG):
    print("=" * 72)
    print("SINDy-PI + PINN  —  Direct 10D  +  Arrhenius exp(-c/T) library")
    print("=" * 72)

    species = cfg['species']
    n_sp    = len(species)
    n_x     = n_sp + 2           # + T + P

    # ── Chemistry setup ───────────────────────────────────────────────────────
    A = build_stoichiometry_matrix(cfg['yaml_path'], species).to(DEVICE)

    nasa9 = parse_nasa9_coeffs(cfg['yaml_path'], species)
    a_lo, a_hi, T_mid = build_nasa9_tensors(nasa9, species, DEVICE)

    # ── Data ──────────────────────────────────────────────────────────────────
    (train_loader, val_loader, test_loader), stats, x_init = load_data(
        cfg['csv_path'], species,
        cfg['val_frac'], cfg['test_frac'], cfg['batch_size']
    )
    x_init = x_init.to(DEVICE)

    # ── Model ─────────────────────────────────────────────────────────────────
    n_exp   = len(cfg['exp_scales'])
    n_theta = library_size(n_x, cfg['sindy_poly_degree'], n_exp)
    model   = SINDyExp(
        n_x         = n_x,
        n_theta     = n_theta,
        n_sp        = n_sp,
        Y_mean      = stats['Y_mean'],
        Y_std       = stats['Y_std'],
        exp_scales  = cfg['exp_scales'],
        poly_degree = cfg['sindy_poly_degree'],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLibrary terms : 1 + {n_x} + {n_x*(n_x+1)//2} (poly) + "
          f"{n_exp} (exp) = {n_theta}")
    print(f"Xi shape      : [{n_theta} × {n_x}]   →   {n_params} learnable parameters")
    print(f"exp(-c/T) c [K]: {cfg['exp_scales']}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
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

        model.train()
        running = dict(total=0.0, sindy=0.0, sparse=0.0, atom=0.0, energy=0.0)
        n_seen  = 0

        for x_norm, xdot_norm, x_raw, u0 in train_loader:
            x_norm    = x_norm.to(DEVICE)
            xdot_norm = xdot_norm.to(DEVICE)
            x_raw     = x_raw.to(DEVICE)
            u0        = u0.to(DEVICE)

            optimizer.zero_grad()

            xdot_pred = model(x_norm)

            L_sindy  = sindy_regression_loss(xdot_pred, xdot_norm)
            L_sparse = sparsity_loss(model.Xi)

            x_init_broadcast = x_init.expand(x_norm.shape[0], -1)
            L_atom = atom_conservation_loss(x_raw, x_init_broadcast, A, n_sp)

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
                'n_sp'             : n_sp,
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

    apply_threshold(model, cfg['sindy_threshold'])

    model.eval()
    test_loss = 0.0; n_test = 0
    with torch.no_grad():
        for x_norm, xdot_norm, x_raw, u0 in test_loader:
            xdot_pred = model(x_norm.to(DEVICE))
            test_loss += F.mse_loss(xdot_pred, xdot_norm.to(DEVICE)).item() * len(x_norm)
            n_test    += len(x_norm)
    print(f"Test SINDy MSE (normalised): {test_loss/n_test:.6e}")

    print_equations(model, n_x, cfg['sindy_poly_degree'], species, cfg['exp_scales'])

    return model, history, stats


# ─────────────────────────────────────────────────────────────────────────────
# 8.  SINDy THRESHOLD  (sequential thresholding)
# ─────────────────────────────────────────────────────────────────────────────
def apply_threshold(model: SINDyExp, threshold: float):
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
def print_equations(model: SINDyExp, n_x: int, degree: int,
                    species: list, exp_scales):
    state_names = [f'log10_{s}' for s in species] + ['log10_T', 'log10_P']
    Xi    = model.Xi.detach().cpu().numpy()
    terms = library_term_names(n_x, degree, state_names, exp_scales)

    print("\n── Discovered SINDy equations  ẋ = Θ(x) Ξ ──")
    for k, sname in enumerate(state_names):
        nz = [(terms[t], Xi[t, k]) for t in range(len(terms)) if abs(Xi[t, k]) > 1e-6]
        if nz:
            eq = ' + '.join(f'{v:+.5f}·{nm}' for nm, v in nz)
        else:
            eq = '0  (zero — fully pruned)'
        print(f"  d({sname})/dt = {eq}")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  SAVE PREDICTIONS CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_predictions_csv(
    model: SINDyExp,
    csv_path: str,
    stats: dict,
    species: list,
    device: str = 'cpu',
    out_path: str = 'sindy_exp_predictions.csv',
) -> pd.DataFrame:
    """Forward-Euler integrate ẋ = Θ(x)Ξ from x(t₀); write prediction CSV."""
    n_sp = len(species)
    df_ref = pd.read_csv(csv_path)

    Y_mean = stats['Y_mean']
    Y_std  = stats['Y_std']

    log10_sp = np.stack([df_ref[f'log10_X_{s}'].values for s in species], axis=1)
    log10_T  = np.log10(df_ref['T_K'].values).reshape(-1, 1)
    log10_P  = np.log10(df_ref['P_Pa'].values).reshape(-1, 1)
    Y_raw    = np.hstack([log10_sp, log10_T, log10_P]).astype(np.float32)

    log10_t = df_ref['log10_t'].values.astype(np.float32)
    dt_log  = np.diff(log10_t, prepend=log10_t[0])

    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)

    model.eval()

    x_int = Y_norm[0:1].copy()
    x_integrated = np.zeros_like(Y_norm)
    x_integrated[0] = x_int

    with torch.no_grad():
        for i in range(1, len(log10_t)):
            xb    = torch.tensor(x_int).to(device)
            dxdt  = model(xb).cpu().numpy()
            x_int = x_int + dxdt * dt_log[i]
            x_int = np.clip(x_int, -16.0, 10.0)
            x_integrated[i] = x_int

    Y_hat_log10 = x_integrated * (Y_std + 1e-8) + Y_mean
    Y_hat_orig  = 10.0 ** Y_hat_log10

    X_pred  = Y_hat_orig[:, :n_sp]
    T_pred  = Y_hat_orig[:, n_sp]
    P_pred  = Y_hat_orig[:, n_sp + 1]

    min_frac     = 1e-16
    log10_X_pred = np.log10(np.clip(X_pred, min_frac, None))

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
def plot_training(history: dict, save_path: str = 'sindy_exp_loss.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(history['train'], label='Train total', lw=2)
    ax.semilogy(history['val'],   label='Val SINDy',   lw=2, ls='--')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title('Training — SINDy-PI + exp(-c/T) library')

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
                    save_path: str = 'sindy_exp_comparison.png'):
    df_gt   = pd.read_csv(csv_path)
    df_pred = pd.read_csv(pred_csv)
    time    = df_gt['time'].values

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_vars = [(f'log10_X_{s}', f'log10_X_{s}', s) for s in species] + \
                [('T_K', 'T_K', 'T [K]')]

    for ax, (gt_col, pred_col, title) in zip(axes, plot_vars):
        gt   = df_gt[gt_col].values
        pred = df_pred[pred_col].values
        ax.semilogx(time, gt,   'k-',  lw=2,   label='Cantera')
        ax.semilogx(time, pred, 'r--', lw=1.5, label='SINDy-PI + exp')
        ax.set_title(title); ax.set_xlabel('t [s]')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle('SINDy-PI + PINN  (with Arrhenius exp(-c/T))  vs  Cantera',
                 fontsize=13)
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
        n_sp_   = ckpt['n_sp']
        n_theta_= ckpt['n_theta']

        model_ = SINDyExp(
            n_x         = n_x_,
            n_theta     = n_theta_,
            n_sp        = n_sp_,
            Y_mean      = stats_['Y_mean'],
            Y_std       = stats_['Y_std'],
            exp_scales  = cfg_['exp_scales'],
            poly_degree = cfg_['sindy_poly_degree'],
        ).to(DEVICE)
        model_.load_state_dict(ckpt['model_state_dict'])

        print_equations(model_, n_x_, cfg_['sindy_poly_degree'],
                        species_, cfg_['exp_scales'])

        save_predictions_csv(
            model_, CFG['csv_path'], stats_, species_,
            device=DEVICE, out_path='sindy_exp_predictions.csv'
        )
        plot_comparison(CFG['csv_path'], 'sindy_exp_predictions.csv', species_)

    else:
        model, history, stats = train(CFG)
        plot_training(history)

        save_predictions_csv(
            model, CFG['csv_path'], stats, CFG['species'],
            device=DEVICE, out_path='sindy_exp_predictions.csv'
        )
        plot_comparison(CFG['csv_path'], 'sindy_exp_predictions.csv', CFG['species'])
