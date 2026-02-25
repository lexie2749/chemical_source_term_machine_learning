import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.01):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.LayerNorm(width)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ChemRelaxNet(nn.Module):
    """
    REDUCED Residual MLP for mapping log10(t) → log10(species + T + P).
    width=64, n_blocks=3  →  ~26,000 parameters
    Requires ~260,000 training samples to satisfy the 10* rule.
    """
    def __init__(self, n_outputs=10, width=64, n_blocks=3, dropout=0.01):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(1, width), nn.LayerNorm(width), nn.SiLU())
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.output_head = nn.Linear(width, n_outputs)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.output_head(self.res_blocks(self.input_proj(x)))


# reduced model 
reduced  = ChemRelaxNet(n_outputs=10, width=64,  n_blocks=3)
print(f'Reduced   (width=64,  3 blocks): {count_params(reduced):>10,} params')
print(f'  → need {count_params(reduced)*10:>10,} training samples (achievable with Cantera)')


def train(
    csv_path:    str   = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data.csv',
    epochs:      int   = 1000,
    batch_size:  int   = 512,
    lr:          float = 3e-4,
    val_frac:    float = 0.1,
    device:      str   = 'auto',
    checkpoint:  str   = 'chem_relax_net.pth',
):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # Extract input and output columns from CSV
    X_raw = df[['log10_t']].values.astype(np.float32)
    
    # Output columns: log10 of species mole fractions, temperature, pressure
    output_col_names = ['log10_X_CO2', 'log10_X_O2', 'log10_X_N2', 'log10_X_CO', 
                        'log10_X_NO', 'log10_X_C', 'log10_X_O', 'log10_X_N']
    Y_raw = df[output_col_names].values.astype(np.float32)
    
    # Add log10 of T and P
    Y_raw = np.hstack([Y_raw, np.log10(df[['T_K', 'P_Pa']].values).astype(np.float32)])

    # Calculate statistics for normalization
    X_mean = X_raw.mean(axis=0).astype(np.float32)
    X_std  = X_raw.std(axis=0).astype(np.float32)
    Y_mean = Y_raw.mean(axis=0).astype(np.float32)
    Y_std  = Y_raw.std(axis=0).astype(np.float32)

    # Normalize
    X_norm = (X_raw - X_mean) / (X_std + 1e-8)
    Y_norm = (Y_raw - Y_mean) / (Y_std + 1e-8)

    X_t = torch.tensor(X_norm)
    Y_t = torch.tensor(Y_norm)

    dataset = TensorDataset(X_t, Y_t)
    n_val   = int(len(dataset) * val_frac)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Train: {n_train} samples  |  Val: {n_val} samples")

    # Store for checkpoint
    output_columns = ['X_CO2', 'X_O2', 'X_N2', 'X_CO', 'X_NO', 'X_C', 'X_O', 'X_N', 'T', 'P']

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ChemRelaxNet(n_outputs=10, width=32, n_blocks=2)  
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    criterion = nn.MSELoss()

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        # — Train —
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * len(xb)
        train_loss = running / n_train

        # — Validate —
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                running += criterion(model(xb), yb).item() * len(xb)
        val_loss = running / n_val

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'X_mean': X_mean, 'X_std': X_std,
                'Y_mean': Y_mean, 'Y_std': Y_std,
                'output_columns': output_columns,
            }, checkpoint)

        if epoch % 100 == 0 or epoch == 1:
            print(f"Epoch {epoch:>5}/{epochs}  "
                  f"train={train_loss:.5f}  val={val_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"\nBest val loss: {best_val_loss:.6f}  → saved to {checkpoint}")
    return model, train_losses, val_losses

# 3.  Inference helper
def predict_batch(model, log10_t: np.ndarray, X_mean, X_std, Y_mean, Y_std, output_columns,
                  device: str = 'cpu') -> dict:
    """
    Predict species mole fractions, temperature, and pressure at given log10(t) values.
    
    Parameters
    ----------
    model : ChemRelaxNet
        Trained model
    log10_t : 1-D array of log10(time) values [seconds]
    X_mean, X_std : normalization stats for input
    Y_mean, Y_std : normalization stats for output
    output_columns : list of output column names
    device : str
        'cpu' or 'cuda'
    
    Returns
    -------
    dict with keys matching output_columns
    """
    # Normalize input
    x = (log10_t.reshape(-1, 1).astype(np.float32) - X_mean) / (X_std + 1e-8)
    x_t = torch.tensor(x).to(device)
    
    model.eval()
    with torch.no_grad():
        y_norm = model(x_t).cpu().numpy()
    
    # De-normalise: convert from normalized space back to log10 space
    y_log10 = y_norm * (Y_std + 1e-8) + Y_mean
    
    # Convert from log10 space to original units
    y_orig = 10.0 ** y_log10
    
    return {name: y_orig[:, i] for i, name in enumerate(output_columns)}


def predict(model, log10_t: np.ndarray, checkpoint_path: str = 'chem_relax_net.pth',
            device: str = 'cpu') -> dict:
    """
    Predict species mole fractions, temperature, and pressure at given times.

    Parameters
    ----------
    log10_t : 1-D array of log10(time) values  [seconds]

    Returns
    -------
    dict with keys: 'CO2','O2','N2','CO','NO','C','O','N','T','P'
                    values are in original units (mole fractions, K, Pa)
    """
    ckpt = torch.load('chem_relax_net.pth', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    X_mean = ckpt['X_mean']
    X_std  = ckpt['X_std']
    Y_mean = ckpt['Y_mean']
    Y_std  = ckpt['Y_std']
    cols   = ckpt['output_columns']

    x = (log10_t.reshape(-1, 1).astype(np.float32) - X_mean) / (X_std + 1e-8)
    x_t = torch.tensor(x).to(device)

    with torch.no_grad():
        y_norm = model(x_t).cpu().numpy()

    # De-normalise
    y_log10 = y_norm * (Y_std + 1e-8) + Y_mean   # log10 of original quantities
    y_orig  = 10.0 ** y_log10                     # back to original units

    return {name: y_orig[:, i] for i, name in enumerate(cols)}

# 4.  Entry point
if __name__ == '__main__':
    csv_path = '/Users/xiaoxizhou/Downloads/adrian_surf/code/training_data.csv'
    
    model, train_losses, val_losses = train(
        csv_path   = csv_path,
        epochs     = 1000,
        batch_size = 512,
        lr         = 3e-4,
    )

    # ── Loss curve ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.semilogy(train_losses, label='Train MSE (norm. log10-space)')
        plt.semilogy(val_losses,   label='Val  MSE (norm. log10-space)')
        plt.xlabel('Epoch')
        plt.ylabel('MSE loss')
        plt.title('ChemRelaxNet Training Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"Could not plot loss curve: {e}")
    
    # ── Generate predictions on training data time points ──────────────────────
    print("\n" + "="*70)
    print("Generating predictions on training data time points...")
    print("="*70)
    
    # Load training data to get time points
    df_train = pd.read_csv(csv_path)
    time_vals = df_train['time'].values
    log10_t_vals = df_train['log10_t'].values
    
    # Load checkpoint to get normalization stats
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load('chem_relax_net.pth', map_location=device)
    X_mean = ckpt['X_mean']
    output_columns = ckpt['output_columns']
    
    # Model predictions
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    predictions = predict_batch(
        model, log10_t_vals, X_mean, X_std, Y_mean, Y_std, 
        output_columns, device=device
    )
    
    # Create output dataframe
    output_data = {
        'time': time_vals,
        'log10_t': log10_t_vals,
    }
    
    # Add predictions
    # The output_columns are ['X_CO2', 'X_O2', 'X_N2', 'X_CO', 'X_NO', 'X_C', 'X_O', 'X_N', 'T', 'P']
    for col_name in output_columns:
        output_data[col_name] = predictions[col_name]
    
    df_output = pd.DataFrame(output_data)
    output_csv_path = 'data_predictions.csv'
    df_output.to_csv(output_csv_path, index=False)
    
    print(f"\n✓ Predictions saved to {output_csv_path}")
    print(f"  Shape: {df_output.shape}")
    print(f"  Columns: {list(df_output.columns)}")
    print(f"\nFirst few rows:")

    print(df_output.head())

    print(df_output.head())

