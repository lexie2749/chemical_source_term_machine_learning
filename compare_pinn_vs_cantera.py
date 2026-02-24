"""
Visualization Script: Compare PINN Predictions vs Cantera Ground Truth

This script:
1. Loads the trained Sequential PINN model
2. Loads the original Cantera simulation data
3. Performs rollout predictions using the PINN
4. Plots both datasets on the same figure for comparison

Usage:
    python compare_pinn_vs_cantera.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

# Updated Import
from sequential_pinn_log import Config, SequentialPINNTrainer 

# This allows PyTorch to safely unpickle your custom Config class
torch.serialization.add_safe_globals([Config])



MODEL_PATH = 'outputs_sequential_log/best_model.pt'
DATA_PATH = 'reaction_data_log.npz'
OUTPUT_DIR = 'outputs_sequential_log'

SPECIES_NAMES = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
COLORS = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink', 'brown']


# ==========================================
# Load Data and Model
# ==========================================
def load_model_and_data():
    """Load trained model and original data."""
    print("=" * 60)
    print("Loading Model and Data")
    print("=" * 60)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nPlease train the model first!")
    
    trainer = SequentialPINNTrainer(config=Config, device=device)
    trainer.load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
    
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found: {DATA_PATH}\nPlease run cantera_pinn_log.py first!")
    
    data = np.load(DATA_PATH)
    print(f"✓ Data loaded from {DATA_PATH}")
    print(f"  Data shape: {data['inputs'].shape}")
    
    return trainer, data, device


# ==========================================
# Reconstruct Time Series from Sequential Data
# ==========================================
def reconstruct_trajectory(data):
    """
    Reconstruct the full trajectory from sequential data.
    
    The data contains pairs: (state_t, dt) -> state_{t+dt}
    We need to reconstruct the full time series.
    """
    inputs = data['inputs']   # (N, 12): [log_X, log_rho, log_T, log_dt]
    outputs = data['outputs'] # (N, 11): [log_X, log_rho, log_T]
    
    # Extract initial state (from first input)
    initial_state_log = inputs[0, :11]  # First 11 elements (everything except log_dt)
    
    # Extract all time steps
    log_dt_array = inputs[:, 11]  # All time steps in log scale
    dt_array = 10 ** log_dt_array  # Convert to linear scale
    
    # Reconstruct time axis
    time_array = np.zeros(len(dt_array) + 1)
    for i in range(len(dt_array)):
        time_array[i+1] = time_array[i] + dt_array[i]
    
    # Reconstruct full trajectory (ground truth from Cantera)
    # First state is initial state, then all outputs
    trajectory_log = np.vstack([initial_state_log, outputs])
    
    print(f"\nReconstructed trajectory:")
    print(f"  Time points: {len(time_array)}")
    print(f"  Time range: [{time_array[0]:.2e}, {time_array[-1]:.2e}] seconds")
    print(f"  Trajectory shape: {trajectory_log.shape}")
    
    return time_array, trajectory_log, log_dt_array


# ==========================================
# PINN Rollout Prediction
# ==========================================
def pinn_rollout(trainer, initial_state_log, log_dt_array):
    """
    Rollout the PINN model for the full trajectory.
    
    Args:
        trainer: Trained PINN model
        initial_state_log: Initial state in log scale (11,)
        log_dt_array: Array of log time steps (N,)
    
    Returns:
        pinn_trajectory: Predicted trajectory (N+1, 11)
    """
    print("\n" + "=" * 60)
    print("PINN Rollout Prediction")
    print("=" * 60)
    
    trajectory = [initial_state_log]
    current_state = initial_state_log.copy()
    
    print(f"Rolling out {len(log_dt_array)} steps...")
    
    for i, log_dt in enumerate(log_dt_array):
        # Predict next state
        next_state = trainer.predict_single_step(current_state, log_dt)
        trajectory.append(next_state)
        current_state = next_state
        
        # Progress indicator
        if (i + 1) % 2000 == 0:
            print(f"  Step {i+1}/{len(log_dt_array)}")
    
    pinn_trajectory = np.array(trajectory)
    print(f"✓ Rollout complete: {pinn_trajectory.shape}")
    
    return pinn_trajectory


# ==========================================
# Compute Errors
# ==========================================
def compute_errors(cantera_traj_log, pinn_traj_log, species_names):
    """Compute prediction errors for each species and temperature."""
    print("\n" + "=" * 60)
    print("Prediction Errors")
    print("=" * 60)
    
    # Errors in log space (more meaningful for concentrations)
    log_errors = np.abs(pinn_traj_log - cantera_traj_log)
    
    print("\nPer-Species Errors (log₁₀ scale - 'decades' of error):")
    for i, name in enumerate(species_names):
        mae = np.mean(log_errors[:, i])
        max_err = np.max(log_errors[:, i])
        print(f"  {name:4s}: MAE = {mae:.3f} decades, Max = {max_err:.3f} decades")
    
    # Temperature error (convert to linear for interpretability)
    T_cantera = 10 ** cantera_traj_log[:, 10]
    T_pinn = 10 ** pinn_traj_log[:, 10]
    T_error = np.abs(T_pinn - T_cantera)
    T_rel_error = (T_error / T_cantera) * 100
    
    print(f"\nTemperature:")
    print(f"  MAE = {np.mean(T_error):.2f} K")
    print(f"  Max Error = {np.max(T_error):.2f} K")
    print(f"  Mean Rel. Error = {np.mean(T_rel_error):.3f}%")
    print(f"  Max Rel. Error = {np.max(T_rel_error):.3f}%")
    
    # Density error
    rho_cantera = 10 ** cantera_traj_log[:, 9]
    rho_pinn = 10 ** pinn_traj_log[:, 9]
    rho_rel_error = np.abs(rho_pinn - rho_cantera) / rho_cantera * 100
    
    print(f"\nDensity:")
    print(f"  Mean Rel. Error = {np.mean(rho_rel_error):.3f}%")
    print(f"  Max Rel. Error = {np.max(rho_rel_error):.3f}%")


# ==========================================
# Plotting Functions
# ==========================================
def plot_species_comparison(time_array, cantera_traj_log, pinn_traj_log, 
                           species_names, colors, output_dir):
    """
    Plot species evolution: Cantera vs PINN.
    
    Reproduces the style from your uploaded figure with both datasets.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, name in enumerate(species_names):
        color = colors[i % len(colors)]
        
        # Cantera ground truth (solid line)
        ax.semilogx(time_array, cantera_traj_log[:, i], 
                   color=color, linewidth=3, alpha=0.7, 
                   linestyle='-', label=f'{name} (Cantera)')
        
        # PINN prediction (dashed line)
        ax.semilogx(time_array, pinn_traj_log[:, i], 
                   color=color, linewidth=2, alpha=0.9,
                   linestyle='--', label=f'{name} (PINN)')
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('log₁₀(Molar Fraction)', fontsize=14, fontweight='bold')
    ax.set_title('Species Evolution: Cantera (solid) vs PINN (dashed)', 
                fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[1], time_array[-1]])  # Skip t=0 for log scale
    
    # Legend with only species names (consolidated)
    handles, labels = ax.get_legend_handles_labels()
    # Take every other handle/label to show only one entry per species
    unique_handles = [handles[i] for i in range(0, len(handles), 2)]
    unique_labels = [species_names[i] for i in range(len(species_names))]
    ax.legend(unique_handles, unique_labels, 
             bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'species_comparison_overlay.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved: {path}")
    plt.show()


def plot_temperature_comparison(time_array, cantera_traj_log, pinn_traj_log, output_dir):
    """Plot temperature evolution: Cantera vs PINN."""
    # Convert to linear scale
    T_cantera = 10 ** cantera_traj_log[:, 10]
    T_pinn = 10 ** pinn_traj_log[:, 10]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogx(time_array, T_cantera, 'k-', linewidth=3, alpha=0.7, label='Cantera')
    ax.semilogx(time_array, T_pinn, 'r--', linewidth=2.5, label='PINN')
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=14, fontweight='bold')
    ax.set_ylabel('Temperature (K)', fontsize=14, fontweight='bold')
    ax.set_title('Temperature Evolution: Cantera vs PINN', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[1], time_array[-1]])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'temperature_comparison.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.show()


def plot_error_evolution(time_array, cantera_traj_log, pinn_traj_log, 
                        species_names, colors, output_dir):
    """Plot how prediction errors evolve over time."""
    # Compute absolute errors in log space
    log_errors = np.abs(pinn_traj_log - cantera_traj_log)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Species errors
    ax = axes[0]
    for i, name in enumerate(species_names):
        color = colors[i % len(colors)]
        ax.semilogx(time_array, log_errors[:, i], 
                   color=color, linewidth=2, alpha=0.8, label=name)
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('|log₁₀(X_PINN) - log₁₀(X_Cantera)|', fontsize=12)
    ax.set_title('Species Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[1], time_array[-1]])
    
    # Temperature error
    ax = axes[1]
    T_cantera = 10 ** cantera_traj_log[:, 10]
    T_pinn = 10 ** pinn_traj_log[:, 10]
    T_error = np.abs(T_pinn - T_cantera)
    
    ax.semilogx(time_array, T_error, 'r-', linewidth=2.5, label='Absolute Error (K)')
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('|T_PINN - T_Cantera| (K)', fontsize=12)
    ax.set_title('Temperature Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[1], time_array[-1]])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_evolution.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.show()


def plot_concentration_parity(cantera_traj_log, pinn_traj_log, species_names, output_dir):
    """Parity plot: predicted vs actual concentrations."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, name in enumerate(species_names):
        ax = axes[i]
        
        # Data in log space
        x = cantera_traj_log[:, i]
        y = pinn_traj_log[:, i]
        
        # Scatter plot with alpha for density visualization
        ax.scatter(x, y, alpha=0.3, s=10, c='blue')
        
        # Perfect prediction line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel(f'Cantera log₁₀({name})', fontsize=10)
        ax.set_ylabel(f'PINN log₁₀({name})', fontsize=10)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
    
    plt.suptitle('Parity Plots: PINN vs Cantera (Log Space)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'parity_plots.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {path}")
    plt.show()


# ==========================================
# Main Function
# ==========================================
def main():
    """Main comparison pipeline."""
    print("\n" + "=" * 60)
    print("PINN vs Cantera Comparison Pipeline")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and data
    trainer, data, device = load_model_and_data()
    
    # Reconstruct Cantera trajectory
    time_array, cantera_traj_log, log_dt_array = reconstruct_trajectory(data)
    
    # Perform PINN rollout
    initial_state_log = cantera_traj_log[0]
    pinn_traj_log = pinn_rollout(trainer, initial_state_log, log_dt_array)
    
    # Compute errors
    compute_errors(cantera_traj_log, pinn_traj_log, SPECIES_NAMES)
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating Comparison Plots")
    print("=" * 60)
    
    plot_species_comparison(time_array, cantera_traj_log, pinn_traj_log,
                          SPECIES_NAMES, COLORS, OUTPUT_DIR)
    
    plot_temperature_comparison(time_array, cantera_traj_log, pinn_traj_log, 
                               OUTPUT_DIR)
    
    plot_error_evolution(time_array, cantera_traj_log, pinn_traj_log,
                        SPECIES_NAMES, COLORS, OUTPUT_DIR)
    
    plot_concentration_parity(cantera_traj_log, pinn_traj_log, 
                             SPECIES_NAMES, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"All plots saved to: {OUTPUT_DIR}/")
    print("\nKey files:")
    print(f"  - species_comparison_overlay.png  (Main comparison plot)")
    print(f"  - temperature_comparison.png      (Temperature evolution)")
    print(f"  - error_evolution.png             (Error over time)")
    print(f"  - parity_plots.png                (Predicted vs actual)")


if __name__ == "__main__":
    main()
