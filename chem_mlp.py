"""
ChemRelaxNet v3
===============
Input  (20-dim):  X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N   (raw,   t0)
                  log10_X_CO2, ..., log10_X_N                       (log10, t0)
                  T_K, P_Pa, rho_kgm3                               (t0)
                  log10(Δt)
Output (11-dim):  X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N   (raw)
                  T_K, P_Pa, rho_kgm3                               at t0 + Δt

CSV schema
----------
time, log10_t,
X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N,
log10_X_CO2, ..., log10_X_N,
T_K, P_Pa, rho_kgm3

Single-trajectory CSV: row 0 is the initial condition.
Split: 80 % train / 10 % val / 10 % test  (time-ordered, no leakage)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────────────────────────────────────
# Model
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
    """
    Residual MLP:  20-dim input  →  11-dim output.
    Default width=128, n_blocks=4.
    """
    N_IN  = 20   # 8 X_raw + 8 log10_X + T + P + rho + log10(Δt)
    N_OUT = 11   # 8 log10_X + T + P + rho  (species in log10 during training)

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


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(df: pd.DataFrame):
    """
    Build (X_raw, Y_raw) from a single-trajectory DataFrame.

    X_raw  shape (N, 20):
        8 X_ic(raw) + 8 log10_X_ic + T_ic + P_ic + rho_ic + log10(Δt)
    Y_raw  shape (N, 11):
        8 log10_X(t) + T(t) + P(t) + rho(t)

    Row 0 is the initial condition; all subsequent rows are targets.
    """
    df = df.sort_values('time').reset_index(drop=True)

    ic = df.iloc[0]
    t0 = float(ic['time'])

    # IC features – fixed for every sample in this trajectory
    ic_feat = np.concatenate([
        ic[SPECIES_COLS].values.astype(np.float32),      # 8 raw X
        ic[LOG_SPECIES_COLS].values.astype(np.float32),  # 8 log10 X
        ic[THERMO_COLS].values.astype(np.float32),       # T, P, rho
    ])  # (19,)

    rows   = df.iloc[1:].reset_index(drop=True)
    dt_arr = rows['time'].values.astype(np.float64) - t0
    valid  = dt_arr > 0
    rows   = rows[valid]
    dt_arr = dt_arr[valid]

    N        = len(rows)
    log10_dt = np.log10(dt_arr).astype(np.float32)[:, None]  # (N, 1)

    X_in = np.tile(ic_feat, (N, 1))                            # (N, 19)
    X_in = np.concatenate([X_in, log10_dt], axis=1)           # (N, 20)

    # Targets: log10 species (from CSV) + raw T, P, rho
    Y_out = np.concatenate([
        rows[LOG_SPECIES_COLS].values.astype(np.float32),     # (N, 8)
        rows[THERMO_COLS].values.astype(np.float32),          # (N, 3)
    ], axis=1)  # (N, 11)

    return X_in, Y_out


def time_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1):
    """
    Time-ordered split for a single trajectory (row 0 = IC, always included).
    Returns (train_df, val_df, test_df).
    """
    df = df.sort_values('time').reset_index(drop=True)
    ic   = df.iloc[[0]]
    rest = df.iloc[1:].reset_index(drop=True)

    n       = len(rest)
    n_test  = max(1, int(n * test_frac))
    n_val   = max(1, int(n * val_frac))
    n_train = n - n_val - n_test

    train_df = pd.concat([ic, rest.iloc[:n_train]],                        ignore_index=True)
    val_df   = pd.concat([ic, rest.iloc[n_train:n_train + n_val]],         ignore_index=True)
    test_df  = pd.concat([ic, rest.iloc[n_train + n_val:]],                ignore_index=True)

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    csv_path:   str   = 'training_data.csv',
    epochs:     int   = 1000,
    batch_size: int   = 512,
    lr:         float = 3e-4,
    val_frac:   float = 0.1,
    test_frac:  float = 0.1,
    width:      int   = 128,
    n_blocks:   int   = 4,
    dropout:    float = 0.01,
    device:     str   = 'auto',
    checkpoint: str   = 'chem_relax_net_v3.pth',
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

    # ── Normalisation (fit on train only) ─────────────────────────────────────
    X_mean = X_train.mean(axis=0).astype(np.float32)
    X_std  = X_train.std(axis=0).astype(np.float32)
    Y_mean = Y_train.mean(axis=0).astype(np.float32)
    Y_std  = Y_train.std(axis=0).astype(np.float32)

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

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # ── Train loop ────────────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # — train —
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * len(xb)
        train_loss = running / len(X_train)

        # — validate —
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * len(xb)
        val_loss = running / len(X_val)

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    return model, train_losses, val_losses, (X_mean, X_std, Y_mean, Y_std)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    species0:     np.ndarray,   # (8,)  raw mole fractions at t0
    log_species0: np.ndarray,   # (8,)  log10 mole fractions at t0
    T0:           float,        # temperature [K]
    P0:           float,        # pressure [Pa]
    rho0:         float,        # density [kg/m3]
    delta_t:      np.ndarray,   # (M,)  time intervals [s]
    checkpoint:   str = 'chem_relax_net_v3.pth',
    device:       str = 'cpu',
) -> dict:
    """
    Predict chemical state at (t0 + delta_t) for a single initial condition.

    Returns
    -------
    dict with keys from OUTPUT_COLS:
      X_CO2, X_O2, X_N2, X_CO, X_NO, X_C, X_O, X_N  – mole fractions
      T_K, P_Pa, rho_kgm3
    Each value is an array of shape (M,).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

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
        np.asarray(species0,     dtype=np.float32),  # 8 raw X
        np.asarray(log_species0, dtype=np.float32),  # 8 log10 X
        [np.float32(T0), np.float32(P0), np.float32(rho0)],
    ])  # (19,)

    X_in = np.tile(ic_feat, (M, 1))                           # (M, 19)
    X_in = np.concatenate([X_in, log10_dt[:, None]], axis=1)  # (M, 20)

    Xn = (X_in - X_mean) / (X_std + 1e-8)
    Xt = torch.tensor(Xn).to(device)

    with torch.no_grad():
        Yn = model(Xt).cpu().numpy()

    Y_denorm = Yn * (Y_std + 1e-8) + Y_mean

    out = {}
    for i, col in enumerate(OUTPUT_COLS):
        if col in SPECIES_COLS:
            out[col] = 10.0 ** Y_denorm[:, i]   # log10 → raw
        else:
            out[col] = Y_denorm[:, i]            # T, P, rho: already raw
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CSV_PATH   = 'training_data.csv'
    CHECKPOINT = 'chem_relax_net_v3.pth'

    model, train_losses, val_losses, norm_stats = train(
        csv_path   = CSV_PATH,
        epochs     = 1000,
        batch_size = 512,
        lr         = 3e-4,
        width      = 128,
        n_blocks   = 4,
        checkpoint = CHECKPOINT,
    )

    # ── Loss curve ────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(train_losses, label='train')
        ax.semilogy(val_losses,   label='val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE loss (normalised)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig('training_loss.png', dpi=150)
        print("Loss curve → training_loss.png")
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
    out_df = pd.DataFrame({
        'time':    times,
        'log10_t': np.log10(times),
        'delta_t': delta_t,
    })
    for col in OUTPUT_COLS:
        out_df[col] = preds[col]
    out_df.to_csv('predictions.csv', index=False)
    print(f"Predictions → predictions.csv  ({len(out_df):,} rows)")
