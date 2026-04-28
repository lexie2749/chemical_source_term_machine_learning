"""
ChemRelaxPINN
=============
Identical architecture and I/O to ChemRelaxNet (chem_mlp.py),
plus three physics-informed loss terms:

  1. Mole-fraction sum  : Σ X_i(t)       = Σ X_i(t0)
  2. Atom conservation  : C / O / N atoms conserved from IC
  3. Energy conservation: Σ X_i·[h_f_i + Cv_i·(T − 298)] = const

Input  (20-dim):  X_CO2…X_N (raw, t0),  log10_X_CO2…log10_X_N (t0),
                  T_K, P_Pa, rho_kgm3 (t0),  log10(Δt)
Output (11-dim):  X_CO2…X_N (raw),  T_K, P_Pa, rho_kgm3  at t0 + Δt

CSV schema: same as training_data.csv (single trajectory, row 0 = IC)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Model  (identical to chem_mlp.py)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class ChemRelaxNet(nn.Module):
    N_IN  = 20
    N_OUT = 11

    def __init__(self, width: int = 128, n_blocks: int = 4, dropout: float = 0.01):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(self.N_IN, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(n_blocks)]
        )
        self.output_head = nn.Linear(width, self.N_OUT)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.res_blocks(self.input_proj(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPECIES_COLS     = ['X_CO2', 'X_O2', 'X_N2', 'X_CO', 'X_NO', 'X_C', 'X_O', 'X_N']
LOG_SPECIES_COLS = ['log10_X_CO2', 'log10_X_O2', 'log10_X_N2', 'log10_X_CO',
                    'log10_X_NO',  'log10_X_C',   'log10_X_O',  'log10_X_N']
THERMO_COLS      = ['T_K', 'P_Pa', 'rho_kgm3']
OUTPUT_COLS      = SPECIES_COLS + THERMO_COLS   # 11 columns

# ── Thermodynamic constants for physics losses ────────────────────────────────
# Species order matches SPECIES_COLS: CO2, O2, N2, CO, NO, C, O, N
R = 8.314   # J/(mol·K)
T_REF = 298.15  # K

# Standard heat of formation h_f [J/mol] at 298 K
H_F = np.array([
    -393509.0,   # CO2
         0.0,   # O2
         0.0,   # N2
    -110535.0,   # CO
     +90291.0,   # NO
    +716682.0,   # C  (monatomic)
    +249173.0,   # O  (monatomic)
    +472683.0,   # N  (monatomic)
], dtype=np.float32)

# Constant-volume heat capacity Cv [J/(mol·K)] (high-T approximation)
#   monatomic : 1.5 R   (C, O, N)
#   diatomic  : 3.5 R   (O2, N2, CO, NO)  – vibrational fully excited at >2000 K
#   CO2       : 4.5 R   (linear triatomic, vibrational fully excited)
CV = np.array([
    4.5 * R,   # CO2
    3.5 * R,   # O2
    3.5 * R,   # N2
    3.5 * R,   # CO
    3.5 * R,   # NO
    1.5 * R,   # C
    1.5 * R,   # O
    1.5 * R,   # N
], dtype=np.float32)

# Atom composition matrix  A[atom, species]
# Atoms: C=0, O=1, N=2
#         CO2  O2  N2  CO  NO   C   O   N
ATOM_MATRIX = np.array([
    [1,   0,   0,   1,   0,   1,   0,   0],   # C
    [2,   2,   0,   1,   1,   0,   1,   0],   # O
    [0,   0,   2,   0,   1,   0,   0,   1],   # N
], dtype=np.float32)   # shape (3, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss helper
# ─────────────────────────────────────────────────────────────────────────────

def _internal_energy(X: torch.Tensor, T: torch.Tensor,
                     h_f: torch.Tensor, cv: torch.Tensor) -> torch.Tensor:
    """
    U = Σ_i  X_i · [h_f_i + Cv_i · (T - T_ref)]   [J/mol-mixture]
    X : (N, 8)  raw mole fractions
    T : (N,)    temperature [K]
    Returns (N,)
    """
    u_i = h_f + cv * (T.unsqueeze(1) - T_REF)   # (N, 8)
    return (X * u_i).sum(dim=1)                   # (N,)


def compute_physics_loss(
    Yn_pred:    torch.Tensor,   # (N, 11)  normalised model output
    Y_mean:     torch.Tensor,   # (11,)
    Y_std:      torch.Tensor,   # (11,)
    ic_sum:     torch.Tensor,   # scalar – Σ X_i at t0
    ic_atoms:   torch.Tensor,   # (3,)    – C/O/N atom counts at t0
    ic_energy:  torch.Tensor,   # scalar  – internal energy at t0 [J/mol]
    h_f:        torch.Tensor,   # (8,)
    cv:         torch.Tensor,   # (8,)
    atom_matrix: torch.Tensor,  # (3, 8)
) -> tuple:
    """
    Returns (loss_sum, loss_atom, loss_energy) — all dimensionless scalars.
    """
    # Denormalise → physical space
    Y_phys  = Yn_pred * (Y_std + 1e-8) + Y_mean       # (N, 11)
    log10_X = Y_phys[:, :8].clamp(min=-30.0)
    X_phys  = 10.0 ** log10_X                          # (N, 8)  raw mole fractions
    T_phys  = Y_phys[:, 8]                             # (N,)    temperature [K]

    # 1. Sum constraint: Σ X_pred ≈ Σ X_ic
    X_sum = X_phys.sum(dim=1)                          # (N,)
    loss_sum = ((X_sum - ic_sum) ** 2).mean()

    # 2. Atom conservation: A @ X_pred ≈ ic_atoms
    #    atom_counts : (N, 3)
    atom_counts = X_phys @ atom_matrix.T               # (N, 3)
    loss_atom   = ((atom_counts - ic_atoms) ** 2).mean()

    # 3. Energy conservation: U_pred ≈ U_ic
    U_pred    = _internal_energy(X_phys, T_phys, h_f, cv)   # (N,)
    loss_energy = ((U_pred - ic_energy) ** 2 /
                   (ic_energy ** 2 + 1e-30)).mean()

    return loss_sum, loss_atom, loss_energy


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers  (identical to chem_mlp.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    df = df.sort_values('time').reset_index(drop=True)
    ic = df.iloc[0]
    t0 = float(ic['time'])

    ic_feat = np.concatenate([
        ic[SPECIES_COLS].values.astype(np.float32),
        ic[LOG_SPECIES_COLS].values.astype(np.float32),
        ic[THERMO_COLS].values.astype(np.float32),
    ])

    rows   = df.iloc[1:].reset_index(drop=True)
    dt_arr = rows['time'].values.astype(np.float64) - t0
    valid  = dt_arr > 0
    rows   = rows[valid]
    dt_arr = dt_arr[valid]

    N        = len(rows)
    log10_dt = np.log10(dt_arr).astype(np.float32)[:, None]

    X_in = np.tile(ic_feat, (N, 1))
    X_in = np.concatenate([X_in, log10_dt], axis=1)

    Y_out = np.concatenate([
        rows[LOG_SPECIES_COLS].values.astype(np.float32),
        rows[THERMO_COLS].values.astype(np.float32),
    ], axis=1)

    return X_in, Y_out


def time_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1):
    df   = df.sort_values('time').reset_index(drop=True)
    ic   = df.iloc[[0]]
    rest = df.iloc[1:].reset_index(drop=True)

    n       = len(rest)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_df = pd.concat([ic, rest.iloc[:n_train]],               ignore_index=True)
    val_df   = pd.concat([ic, rest.iloc[n_train:n_train+n_val]],  ignore_index=True)
    test_df  = pd.concat([ic, rest.iloc[n_train+n_val:]],         ignore_index=True)
    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    csv_path:       str   = 'training_data.csv',
    epochs:         int   = 1000,
    batch_size:     int   = 512,
    lr:             float = 3e-4,
    val_frac:       float = 0.1,
    test_frac:      float = 0.1,
    width:          int   = 128,
    n_blocks:       int   = 4,
    dropout:        float = 0.01,
    device:         str   = 'auto',
    checkpoint:     str   = 'chem_relax_pinn.pth',
    lambda_sum:     float = 1.0,
    lambda_atom:    float = 1.0,
    lambda_energy:  float = 1e-10,   # energy is in J²/mol² → needs small weight
):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Load & split ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path).sort_values('time').reset_index(drop=True)
    print(f"Loaded {len(df):,} rows.  "
          f"Time: {df['time'].min():.3e} → {df['time'].max():.3e} s")

    train_df, val_df, test_df = time_split(df, val_frac, test_frac)
    print(f"Samples → train: {len(train_df)-1:,}  "
          f"val: {len(val_df)-1:,}  "
          f"test: {len(test_df)-1:,}")

    # ── Build (X, Y) arrays ───────────────────────────────────────────────────
    X_train, Y_train = build_dataset(train_df)
    X_val,   Y_val   = build_dataset(val_df)
    X_test,  Y_test  = build_dataset(test_df)
    print(f"Feature shapes: X={X_train.shape}  Y={Y_train.shape}")

    # ── Precompute IC physics quantities ──────────────────────────────────────
    ic        = df.iloc[0]
    X_ic      = ic[SPECIES_COLS].values.astype(np.float32)   # (8,)
    T_ic      = float(ic['T_K'])

    ic_sum    = float(X_ic.sum())
    ic_atoms  = (ATOM_MATRIX @ X_ic).astype(np.float32)      # (3,)
    ic_energy = float((X_ic * (H_F + CV * (T_ic - T_REF))).sum())

    print(f"IC  sum={ic_sum:.4f}  "
          f"atoms=[C:{ic_atoms[0]:.4f}, O:{ic_atoms[1]:.4f}, N:{ic_atoms[2]:.4f}]  "
          f"U={ic_energy/1e3:.2f} kJ/mol")

    # Convert IC quantities to device tensors
    t_ic_sum    = torch.tensor(ic_sum,    dtype=torch.float32, device=device)
    t_ic_atoms  = torch.tensor(ic_atoms,  dtype=torch.float32, device=device)
    t_ic_energy = torch.tensor(ic_energy, dtype=torch.float32, device=device)
    t_h_f       = torch.tensor(H_F,       dtype=torch.float32, device=device)
    t_cv        = torch.tensor(CV,        dtype=torch.float32, device=device)
    t_atom_mat  = torch.tensor(ATOM_MATRIX, dtype=torch.float32, device=device)

    # ── Normalisation ─────────────────────────────────────────────────────────
    X_mean = X_train.mean(axis=0).astype(np.float32)
    X_std  = X_train.std(axis=0).astype(np.float32)
    Y_mean = Y_train.mean(axis=0).astype(np.float32)
    Y_std  = Y_train.std(axis=0).astype(np.float32)

    t_Y_mean = torch.tensor(Y_mean, device=device)
    t_Y_std  = torch.tensor(Y_std,  device=device)

    def normalise(X, Y):
        Xn = (X - X_mean) / (X_std + 1e-8)
        Yn = (Y - Y_mean) / (Y_std + 1e-8)
        return torch.tensor(Xn), torch.tensor(Yn)

    Xt_tr, Yt_tr = normalise(X_train, Y_train)
    Xt_va, Yt_va = normalise(X_val,   Y_val)
    Xt_te, Yt_te = normalise(X_test,  Y_test)

    def make_loader(Xt, Yt, shuffle):
        return DataLoader(TensorDataset(Xt, Yt),
                          batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    train_loader = make_loader(Xt_tr, Yt_tr, shuffle=True)
    val_loader   = make_loader(Xt_va, Yt_va, shuffle=False)
    test_loader  = make_loader(Xt_te, Yt_te, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ChemRelaxNet(width=width, n_blocks=n_blocks, dropout=dropout).to(device)
    print(f"Parameters: {count_params(model):,}")
    print(f"λ_sum={lambda_sum}  λ_atom={lambda_atom}  λ_energy={lambda_energy:.2e}")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # ── Train loop ────────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    history = {'train': [], 'val': [],
               'loss_sum': [], 'loss_atom': [], 'loss_energy': []}

    for epoch in range(1, epochs + 1):
        # — train —
        model.train()
        run_total = run_data = run_sum = run_atom = run_energy = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()

            Yn_pred = model(xb)

            

            # Physics losses (physical space)
            l_sum, l_atom, l_energy = compute_physics_loss(
                Yn_pred, t_Y_mean, t_Y_std,
                t_ic_sum, t_ic_atoms, t_ic_energy,
                t_h_f, t_cv, t_atom_mat,
            )

            loss = (
                    + lambda_sum    * l_sum
                    + lambda_atom   * l_atom
                    + lambda_energy * l_energy)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            n = len(xb)
            run_total  += loss.item()      * n
            run_sum    += l_sum.item()     * n
            run_atom   += l_atom.item()    * n
            run_energy += l_energy.item()  * n

        N_tr = len(X_train)
        train_loss = run_total / N_tr

        # — validate (data loss only) —
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * len(xb)
        val_loss = running / len(X_val)

        scheduler.step()
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['loss_sum'].append(run_sum    / N_tr)
        history['loss_atom'].append(run_atom   / N_tr)
        history['loss_energy'].append(run_energy / N_tr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'val_loss':         val_loss,
                'width':            width,
                'n_blocks':         n_blocks,
                'X_mean': X_mean,   'X_std': X_std,
                'Y_mean': Y_mean,   'Y_std': Y_std,
                'output_columns':   OUTPUT_COLS,
            }, checkpoint)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:>5}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"sum={run_sum/N_tr:.3e}  atom={run_atom/N_tr:.3e}  "
                  f"energy={run_energy/N_tr:.3e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nBest val loss: {best_val_loss:.6f}  → saved to '{checkpoint}'")

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            running += criterion(model(xb), yb).item() * len(xb)
    test_loss = running / len(X_test)
    print(f"Test  MSE (normalised): {test_loss:.6f}")

    return model, history, (X_mean, X_std, Y_mean, Y_std)


# ─────────────────────────────────────────────────────────────────────────────
# Inference  (identical to chem_mlp.py)
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    species0:     np.ndarray,
    log_species0: np.ndarray,
    T0:           float,
    P0:           float,
    rho0:         float,
    delta_t:      np.ndarray,
    checkpoint:   str = 'chem_relax_pinn.pth',
    device:       str = 'cpu',
) -> dict:
    """Same interface as chem_mlp.predict()."""
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ChemRelaxNet(width=ckpt['width'], n_blocks=ckpt['n_blocks']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X_mean = ckpt['X_mean']
    X_std  = ckpt['X_std']
    Y_mean = ckpt['Y_mean']
    Y_std  = ckpt['Y_std']

    M        = len(delta_t)
    log10_dt = np.log10(np.maximum(delta_t, 1e-300)).astype(np.float32)

    ic_feat = np.concatenate([
        np.asarray(species0,     dtype=np.float32),
        np.asarray(log_species0, dtype=np.float32),
        [np.float32(T0), np.float32(P0), np.float32(rho0)],
    ])

    X_in = np.tile(ic_feat, (M, 1))
    X_in = np.concatenate([X_in, log10_dt[:, None]], axis=1)
    Xn   = (X_in - X_mean) / (X_std + 1e-8)
    Xt   = torch.tensor(Xn).to(device)

    with torch.no_grad():
        Yn = model(Xt).cpu().numpy()

    Y_denorm = Yn * (Y_std + 1e-8) + Y_mean

    out = {}
    for i, col in enumerate(OUTPUT_COLS):
        if col in SPECIES_COLS:
            out[col] = 10.0 ** Y_denorm[:, i]
        else:
            out[col] = Y_denorm[:, i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CSV_PATH   = 'training_data.csv'
    CHECKPOINT = 'chem_relax_pinn.pth'

    model, history, norm_stats = train(
        csv_path       = CSV_PATH,
        epochs         = 1000,
        batch_size     = 512,
        lr             = 3e-4,
        width          = 128,
        n_blocks       = 4,
        checkpoint     = CHECKPOINT,
        lambda_sum     = 1.0,
        lambda_atom    = 1.0,
        lambda_energy  = 1e-10,
    )

    # ── Loss curve ────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        ax = axes[0]
        ax.semilogy(history['train'], label='train (total)')
        ax.semilogy(history['val'],   label='val (data)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total & Data Loss')
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        ax.semilogy(history['loss_sum'],    label='sum')
        ax.semilogy(history['loss_atom'],   label='atom')
        ax.semilogy(history['loss_energy'], label='energy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss')
        ax.set_title('Physics Losses')
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        fig.savefig('pinn_training_loss.png', dpi=150)
        print("Loss curve → pinn_training_loss.png")
    except ImportError:
        pass

    # ── Inference on all training time steps ─────────────────────────────────
    df = pd.read_csv(CSV_PATH).sort_values('time').reset_index(drop=True)
    ic = df.iloc[0]

    species0     = ic[SPECIES_COLS].values
    log_species0 = ic[LOG_SPECIES_COLS].values
    T0, P0, rho0 = float(ic['T_K']), float(ic['P_Pa']), float(ic['rho_kgm3'])

    t0      = float(ic['time'])
    delta_t = df['time'].values[1:] - t0
    delta_t = delta_t[delta_t > 0]

    preds = predict(species0, log_species0, T0, P0, rho0, delta_t,
                    checkpoint=CHECKPOINT)

    times  = delta_t + t0
    out_df = pd.DataFrame({'time': times, 'log10_t': np.log10(times), 'delta_t': delta_t})
    for col in OUTPUT_COLS:
        out_df[col] = preds[col]
    out_df.to_csv('pinn_predictions.csv', index=False)
    print(f"Predictions → pinn_predictions.csv  ({len(out_df):,} rows)")
