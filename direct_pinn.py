"""
Direct Prediction PINN for Chemical Kinetics

This PINN directly predicts species concentrations at any target time t,
eliminating error accumulation from step-by-step rollout.

Architecture:
    Input:  [T_0_normalized, log10(t)]  -> 2 values
    Output: [X1, ..., X9, rho, T]       -> 11 values

Usage:
    python direct_pinn.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
import time as timer

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# Configuration
# ==========================================
class Config:
    """Centralized configuration for the Direct PINN model."""
    
    # Model architecture
    INPUT_DIM = 2    # [T_0_normalized, log10(t)]
    OUTPUT_DIM = 11  # [X1, ..., X9, rho, T]
    HIDDEN_DIMS = [256, 256, 128]  # Hidden layer sizes
    
    # Normalization scales (must match data generation)
    SCALE_T = 8000.0
    SCALE_RHO = 0.002
    
    # Training parameters
    EPOCHS = 5000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    PRINT_EVERY = 500
    
    # Loss weights
    LAMBDA_DATA = 1.0       # Data fitting loss
    LAMBDA_SUM = 10.0       # Species sum to 1
    LAMBDA_NONNEG = 5.0     # Non-negativity
    LAMBDA_ATOM = 1.0       # Atom conservation
    
    # Paths
    DATA_PATH = 'reaction_data_direct.npz'
    MODEL_PATH = 'direct_pinn_model.pt'
    OUTPUT_DIR = 'outputs_direct'
    
    # Species information
    SPECIES_NAMES = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
    N_SPECIES = 9
    
    # Molar masses (kg/mol)
    MOLAR_MASSES = {
        'CO2': 0.04401, 'O2': 0.032, 'N2': 0.028014,
        'CO': 0.02801, 'NO': 0.03001, 'C': 0.01201,
        'O': 0.016, 'N': 0.014007, 'AR': 0.039948
    }
    
    # Atomic composition for conservation laws
    ATOMIC_COMPOSITION = {
        'CO2': {'C': 1, 'O': 2},
        'O2': {'O': 2},
        'N2': {'N': 2},
        'CO': {'C': 1, 'O': 1},
        'NO': {'N': 1, 'O': 1},
        'C': {'C': 1},
        'O': {'O': 1},
        'N': {'N': 1},
        'AR': {'AR': 1}
    }


# ==========================================
# Neural Network Architecture
# ==========================================
class DirectPINN(nn.Module):
    """
    Direct Prediction Physics-Informed Neural Network.
    
    Maps (T_0_normalized, log10(t)) -> (X1, ..., X9, rho, T)
    """
    
    def __init__(self, input_dim=2, output_dim=11, hidden_dims=[256, 256, 128]):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Smooth activation for interpolation
            prev_dim = hidden_dim
        
        # Output layer (no activation - raw outputs)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)
    
    def predict_with_constraints(self, x):
        """
        Forward pass with physical constraints applied.
        - Softmax on species to ensure sum = 1 and non-negativity
        """
        raw_output = self.forward(x)
        
        # Apply softmax to species (first 9 outputs)
        X_pred = torch.softmax(raw_output[:, :9], dim=1)
        
        # Keep rho and T as-is (they're normalized)
        rho_T_pred = raw_output[:, 9:]
        
        return torch.cat([X_pred, rho_T_pred], dim=1)


# ==========================================
# Physics-Informed Loss Functions
# ==========================================
class PhysicsLoss:
    """Physics-based loss terms for chemical reactions."""
    
    def __init__(self, config=Config):
        self.config = config
        self.n_species = config.N_SPECIES
        self.species_names = config.SPECIES_NAMES
        self.molar_masses = config.MOLAR_MASSES
        self.atomic_composition = config.ATOMIC_COMPOSITION
    
    def sum_of_fractions_loss(self, X_pred):
        """
        Molar fractions must sum to 1: Σ X_i = 1
        
        Note: If using softmax, this is automatically satisfied.
        This loss is for when NOT using softmax.
        """
        sum_X = torch.sum(X_pred[:, :self.n_species], dim=1)
        return torch.mean((sum_X - 1.0) ** 2)
    
    def non_negativity_loss(self, X_pred):
        """
        Molar fractions must be non-negative: X_i >= 0
        
        Note: If using softmax, this is automatically satisfied.
        """
        return torch.mean(torch.relu(-X_pred[:, :self.n_species]) ** 2)
    
    def atom_conservation_loss(self, X_pred, X_true):
        """
        Conservation of atoms between prediction and ground truth.
        
        For direct prediction from the same initial condition,
        atom counts should match the ground truth.
        """
        batch_size = X_pred.shape[0]
        device = X_pred.device
        
        atom_names = ['C', 'O', 'N', 'AR']
        losses = []
        
        for atom in atom_names:
            # Count atoms in prediction
            atom_count_pred = torch.zeros(batch_size, device=device)
            atom_count_true = torch.zeros(batch_size, device=device)
            
            for i, species in enumerate(self.species_names):
                n_atoms = self.atomic_composition.get(species, {}).get(atom, 0)
                atom_count_pred += n_atoms * X_pred[:, i]
                atom_count_true += n_atoms * X_true[:, i]
            
            # Atom counts should match
            loss = torch.mean((atom_count_pred - atom_count_true) ** 2)
            losses.append(loss)
        
        return sum(losses) / len(losses)
    
    def smoothness_loss(self, model, inputs, epsilon=0.01):
        """
        Encourage smooth predictions in log-time space.
        Predictions at nearby times should be similar.
        
        This is optional and helps with interpolation.
        """
        # Perturb the log10(t) input slightly
        inputs_perturbed = inputs.clone()
        inputs_perturbed[:, 1] += epsilon * torch.randn_like(inputs_perturbed[:, 1])
        
        # Get predictions at both points
        pred_original = model(inputs)
        pred_perturbed = model(inputs_perturbed)
        
        # Penalize large differences
        return torch.mean((pred_original - pred_perturbed) ** 2)


# ==========================================
# Training Framework
# ==========================================
class DirectPINNTrainer:
    """Complete training framework for Direct Prediction PINN."""
    
    def __init__(self, config=Config, device='cpu'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = DirectPINN(
            input_dim=config.INPUT_DIM,
            output_dim=config.OUTPUT_DIM,
            hidden_dims=config.HIDDEN_DIMS
        ).to(device)
        
        # Create physics loss calculator
        self.physics_loss = PhysicsLoss(config)
        
        # Loss history
        self.loss_history = {
            'total': [], 'data': [], 'physics': [],
            'sum': [], 'nonneg': [], 'atom': []
        }
        
        # Training stats
        self.best_loss = float('inf')
        self.best_state = None
    
    def compute_loss(self, inputs, targets, use_softmax=True):
        """
        Compute total loss: L = L_data + λ * L_physics
        
        Args:
            inputs: [T_0_norm, log10(t)] - shape (batch, 2)
            targets: [X1..X9, rho_norm, T_norm] - shape (batch, 11)
            use_softmax: Whether to apply softmax to species outputs
        """
        # Forward pass
        if use_softmax:
            outputs = self.model.predict_with_constraints(inputs)
        else:
            outputs = self.model(inputs)
        
        # Extract components
        X_pred = outputs[:, :self.config.N_SPECIES]
        X_true = targets[:, :self.config.N_SPECIES]
        
        # Data loss (MSE)
        loss_data = torch.mean((outputs - targets) ** 2)
        
        # Physics losses
        if use_softmax:
            # Softmax already ensures sum=1 and non-negativity
            loss_sum = torch.tensor(0.0, device=self.device)
            loss_nonneg = torch.tensor(0.0, device=self.device)
        else:
            loss_sum = self.physics_loss.sum_of_fractions_loss(X_pred)
            loss_nonneg = self.physics_loss.non_negativity_loss(X_pred)
        
        # Atom conservation
        loss_atom = self.physics_loss.atom_conservation_loss(X_pred, X_true)
        
        # Total physics loss
        loss_physics = (
            self.config.LAMBDA_SUM * loss_sum +
            self.config.LAMBDA_NONNEG * loss_nonneg +
            self.config.LAMBDA_ATOM * loss_atom
        )
        
        # Total loss
        loss_total = self.config.LAMBDA_DATA * loss_data + loss_physics
        
        losses = {
            'total': loss_total.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'sum': loss_sum.item(),
            'nonneg': loss_nonneg.item(),
            'atom': loss_atom.item()
        }
        
        return loss_total, losses, outputs
    
    def train(self, train_loader, val_loader=None, epochs=None, learning_rate=None):
        """Train the Direct PINN."""
        
        epochs = epochs or self.config.EPOCHS
        learning_rate = learning_rate or self.config.LEARNING_RATE
        
        # Optimizer and scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=300
        )
        
        print("\n" + "=" * 60)
        print("Direct Prediction PINN - Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {learning_rate}")
        print("-" * 60)
        
        start_time = timer.time()
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {key: 0.0 for key in self.loss_history.keys()}
            n_batches = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Compute loss
                loss, losses, _ = self.compute_loss(inputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value
                n_batches += 1
            
            # Average losses
            for key in epoch_losses.keys():
                epoch_losses[key] /= n_batches
                self.loss_history[key].append(epoch_losses[key])
            
            # Learning rate scheduling
            scheduler.step(epoch_losses['total'])
            
            # Track best model
            if epoch_losses['total'] < self.best_loss:
                self.best_loss = epoch_losses['total']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # Print progress
            if (epoch + 1) % self.config.PRINT_EVERY == 0 or epoch == 0:
                elapsed = timer.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1:5d}/{epochs} | Time: {elapsed:.1f}s | LR: {lr:.2e}")
                print(f"  Total: {epoch_losses['total']:.6e} | "
                      f"Data: {epoch_losses['data']:.6e} | "
                      f"Atom: {epoch_losses['atom']:.6e}")
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        print("-" * 60)
        print(f"Training complete! Best loss: {self.best_loss:.6e}")
        print(f"Total time: {timer.time() - start_time:.1f}s")
        
        return self.loss_history
    
    def predict(self, log10_t, T_0=7500.0):
        """
        Predict state at given time(s).
        
        Args:
            log10_t: log10 of target time (scalar or array)
            T_0: Initial temperature (default: 7500 K)
        
        Returns:
            Dictionary with species, rho, T predictions
        """
        self.model.eval()
        
        # Handle scalar input
        if np.isscalar(log10_t):
            log10_t = np.array([log10_t])
        log10_t = np.atleast_1d(log10_t)
        
        # Create input tensor
        n_points = len(log10_t)
        inputs = np.zeros((n_points, 2), dtype=np.float32)
        inputs[:, 0] = T_0 / self.config.SCALE_T
        inputs[:, 1] = log10_t
        
        inputs_tensor = torch.FloatTensor(inputs).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.predict_with_constraints(inputs_tensor)
        
        outputs_np = outputs.cpu().numpy()
        
        # Denormalize
        X_pred = outputs_np[:, :9]
        rho_pred = outputs_np[:, 9] * self.config.SCALE_RHO
        T_pred = outputs_np[:, 10] * self.config.SCALE_T
        
        return {
            'X': X_pred,
            'rho': rho_pred,
            'T': T_pred,
            'species_names': self.config.SPECIES_NAMES
        }
    
    def save_model(self, path=None):
        """Save model checkpoint."""
        path = path or self.config.MODEL_PATH
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'input_dim': self.config.INPUT_DIM,
                'output_dim': self.config.OUTPUT_DIM,
                'hidden_dims': self.config.HIDDEN_DIMS,
            },
            'loss_history': self.loss_history,
            'best_loss': self.best_loss
        }, path)
        print(f"Model saved to: {path}")
    
    def load_model(self, path=None):
        """Load model from checkpoint."""
        path = path or self.config.MODEL_PATH
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Model loaded from: {path}")


# ==========================================
# Data Loading
# ==========================================
def load_data(data_path=Config.DATA_PATH, train_split=0.8, batch_size=Config.BATCH_SIZE):
    """
    Load and prepare data for training.
    
    Returns:
        train_loader, val_loader, raw_data
    """
    print(f"\nLoading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run 'python generate_direct_data.py' first."
        )
    
    data = np.load(data_path)
    inputs = data['inputs']    # Shape: (N, 2) - [T_0_norm, log10(t)]
    outputs = data['outputs']  # Shape: (N, 11) - [X1..X9, rho, T]
    time_array = data['time']
    
    print(f"  Loaded {len(inputs)} samples")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    
    # Normalize outputs (species are already 0-1, normalize rho and T)
    outputs_norm = outputs.copy()
    outputs_norm[:, 9] /= Config.SCALE_RHO   # rho
    outputs_norm[:, 10] /= Config.SCALE_T    # T
    
    # Split into train/val
    n_samples = len(inputs)
    n_train = int(n_samples * train_split)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(inputs[train_idx]),
        torch.FloatTensor(outputs_norm[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(inputs[val_idx]),
        torch.FloatTensor(outputs_norm[val_idx])
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    raw_data = {
        'inputs': inputs,
        'outputs': outputs,
        'outputs_norm': outputs_norm,
        'time': time_array,
        'species_names': list(data['species_names'])
    }
    
    return train_loader, val_loader, raw_data


# ==========================================
# Evaluation and Visualization
# ==========================================
def evaluate_and_plot(trainer, raw_data, output_dir=Config.OUTPUT_DIR):
    """
    Evaluate model and generate comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Evaluation and Visualization")
    print("=" * 60)
    
    # Get ground truth
    time_array = raw_data['time']
    outputs_true = raw_data['outputs']
    X_true = outputs_true[:, :9]
    T_true = outputs_true[:, 10]
    rho_true = outputs_true[:, 9]
    
    # Get predictions at same time points
    log10_t = np.log10(time_array)
    predictions = trainer.predict(log10_t)
    X_pred = predictions['X']
    T_pred = predictions['T']
    rho_pred = predictions['rho']
    
    # Compute errors
    species_names = Config.SPECIES_NAMES
    
    print("\nPer-Species Errors (log10 scale):")
    for i, name in enumerate(species_names):
        # Compute error in log space (more meaningful for chemistry)
        log_true = np.log10(np.maximum(X_true[:, i], 1e-20))
        log_pred = np.log10(np.maximum(X_pred[:, i], 1e-20))
        mae = np.mean(np.abs(log_pred - log_true))
        print(f"  {name:4s}: MAE = {mae:.4f} (log10 scale)")
    
    temp_rmse = np.sqrt(np.mean((T_pred - T_true) ** 2))
    temp_rel_err = np.mean(np.abs(T_pred - T_true) / T_true) * 100
    print(f"\nTemperature: RMSE = {temp_rmse:.2f} K, Rel. Error = {temp_rel_err:.2f}%")
    
    # ==========================================
    # Plot 1: Species Evolution Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['r', 'g', 'k', 'm', 'c', 'orange', 'purple', 'pink', 'brown']
    
    for i, name in enumerate(species_names):
        # Ground truth (solid line)
        log_true = np.log10(np.maximum(X_true[:, i], 1e-20))
        ax.semilogx(time_array, log_true, '-', color=colors[i], 
                   linewidth=2.5, alpha=0.6)
        
        # Prediction (dashed line)
        log_pred = np.log10(np.maximum(X_pred[:, i], 1e-20))
        ax.semilogx(time_array, log_pred, '--', color=colors[i], 
                   linewidth=2, label=name)
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('log₁₀(Molar Fraction)', fontsize=12)
    ax.set_title('Species Evolution: Cantera (solid) vs PINN (dashed)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[0], time_array[-1]])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'species_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.show()
    
    # ==========================================
    # Plot 2: Temperature Comparison
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.semilogx(time_array, T_true, 'k-', linewidth=3, alpha=0.6, label='Cantera')
    ax.semilogx(time_array, T_pred, 'r--', linewidth=2.5, label='PINN')
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('Temperature (K)', fontsize=12)
    ax.set_title('Temperature Evolution: Cantera vs PINN', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[0], time_array[-1]])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'temperature_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()
    
    # ==========================================
    # Plot 3: Training Loss History
    # ==========================================
    if trainer.loss_history['total']:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Total and data loss
        ax = axes[0]
        ax.semilogy(trainer.loss_history['total'], label='Total', linewidth=2)
        ax.semilogy(trainer.loss_history['data'], label='Data', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Physics loss
        ax = axes[1]
        ax.semilogy(trainer.loss_history['atom'], label='Atom Conservation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Physics Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, 'training_loss.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.show()
    
    # ==========================================
    # Plot 4: Prediction Error Over Time
    # ==========================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Temperature error
    temp_error = np.abs(T_pred - T_true)
    ax.semilogx(time_array, temp_error, 'r-', linewidth=2, label='Temperature Error (K)')
    
    ax.set_xlabel('Time (s) - Log Scale', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([time_array[0], time_array[-1]])
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'error_over_time.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.show()
    
    return {
        'X_pred': X_pred,
        'X_true': X_true,
        'T_pred': T_pred,
        'T_true': T_true,
        'time': time_array
    }


# ==========================================
# Main Entry Point
# ==========================================
def main():
    """Main function to run the complete Direct PINN pipeline."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("Direct Prediction PINN for Chemical Kinetics")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader, raw_data = load_data()
    
    # Create trainer
    trainer = DirectPINNTrainer(config=Config, device=device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save model
    model_path = os.path.join(Config.OUTPUT_DIR, Config.MODEL_PATH)
    trainer.save_model(model_path)
    
    # Evaluate and plot
    results = evaluate_and_plot(trainer, raw_data)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {Config.OUTPUT_DIR}/")
    
    # Demo: Query at specific times
    print("\n--- Demo: Query at specific times ---")
    demo_times = [1e-10, 1e-6, 1e-3, 1.0]
    for t in demo_times:
        pred = trainer.predict(np.log10(t))
        print(f"t = {t:.0e} s: T = {pred['T'][0]:.1f} K, "
              f"X_CO2 = {pred['X'][0, 0]:.4f}, X_CO = {pred['X'][0, 3]:.4f}")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
