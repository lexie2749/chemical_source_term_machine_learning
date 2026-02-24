"""
Sequential PINN for Chemical Kinetics with Logarithmic Inputs

This PINN predicts the next state given current state and time step,
ALL IN LOGARITHMIC SCALE for better handling of vastly different scales.

Architecture:
    Input:  [log10(X1), ..., log10(X9), log10(rho), log10(T), log10(dt)] -> 12 values
    Output: [log10(X1), ..., log10(X9), log10(rho), log10(T)]           -> 11 values

Key advantages:
    - log10(dt=1e-14) = -14.0 vs log10(dt=1.0) = 0.0 → meaningful differences
    - log10(X=1e-16) = -16.0 vs log10(X=1.0) = 0.0 → captures trace species
    - Better gradient flow and convergence for neural networks

Usage:
    python sequential_pinn_log.py
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
    """Centralized configuration for the Sequential PINN model."""
    
    # Model architecture
    INPUT_DIM = 12   # [log10(X1-X9), log10(rho), log10(T), log10(dt)]
    OUTPUT_DIM = 11  # [log10(X1-X9), log10(rho), log10(T)]
    HIDDEN_DIMS = [256, 256, 256, 128]  # Deeper network for complex dynamics
    
    # Training parameters
    EPOCHS = 3000
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 512
    PRINT_EVERY = 200
    
    # Loss weights
    LAMBDA_DATA = 1.0           # Data fitting loss
    LAMBDA_ATOM = 5.0           # Atom conservation (in linear space)
    LAMBDA_MASS = 5.0           # Mass conservation
    LAMBDA_MONOTONICITY = 0.1   # Temperature monotonicity for equilibration
    
    # Paths
    DATA_PATH = 'reaction_data_log.npz'
    MODEL_PATH = 'sequential_pinn_log.pt'
    OUTPUT_DIR = 'outputs_sequential_log'
    
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
    
    # Minimum values for log transformation (must match data generation)
    MIN_CONC = 1e-20
    MIN_RHO = 1e-6
    MIN_T = 1.0
    MIN_DT = 1e-20


# ==========================================
# Neural Network Architecture
# ==========================================
class SequentialPINNLog(nn.Module):
    """
    Sequential Physics-Informed Neural Network with Logarithmic Inputs.
    
    Maps [log10(X), log10(rho), log10(T), log10(dt)] -> [log10(X'), log10(rho'), log10(T')]
    """
    
    def __init__(self, input_dim=12, output_dim=11, hidden_dims=[256, 256, 256, 128]):
        super().__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())  # Smooth activation for physical systems
            prev_dim = hidden_dim
        
        # Output layer (no activation - direct log outputs)
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 12) containing log-scale features
        
        Returns:
            Output tensor of shape (batch, 11) containing log-scale predictions
        """
        return self.network(x)


# ==========================================
# Physics-Informed Loss Functions
# ==========================================
class PhysicsLoss:
    """Physics-based loss terms for chemical reactions in log space."""
    
    def __init__(self, config=Config):
        self.config = config
        self.n_species = config.N_SPECIES
        self.species_names = config.SPECIES_NAMES
        self.molar_masses = config.MOLAR_MASSES
        self.atomic_composition = config.ATOMIC_COMPOSITION
    
    def atom_conservation_loss(self, pred_log, true_log):
        """
        Conservation of atoms between prediction and ground truth.
        
        This is computed in LINEAR space (10^log_X) to ensure physical meaning.
        Atom counts must be conserved: Σ n_i * X_i = constant
        
        Args:
            pred_log: Predicted log concentrations (batch, 11)
            true_log: True log concentrations (batch, 11)
        """
        batch_size = pred_log.shape[0]
        device = pred_log.device
        
        # Convert back to linear space for atom counting
        X_pred = torch.pow(10, pred_log[:, :self.n_species])  # (batch, 9)
        X_true = torch.pow(10, true_log[:, :self.n_species])  # (batch, 9)
        
        atom_names = ['C', 'O', 'N', 'AR']
        losses = []
        
        for atom in atom_names:
            # Count atoms in prediction and ground truth
            atom_count_pred = torch.zeros(batch_size, device=device)
            atom_count_true = torch.zeros(batch_size, device=device)
            
            for i, species in enumerate(self.species_names):
                n_atoms = self.atomic_composition.get(species, {}).get(atom, 0)
                atom_count_pred += n_atoms * X_pred[:, i]
                atom_count_true += n_atoms * X_true[:, i]
            
            # Relative error in atom counts
            loss = torch.mean(((atom_count_pred - atom_count_true) / 
                              (atom_count_true + 1e-10)) ** 2)
            losses.append(loss)
        
        return sum(losses) / len(losses)
    
    def mass_conservation_loss(self, pred_log, true_log):
        """
        Total mass conservation: ρ * V = constant
        
        For constant volume reactions, density should be related to
        total molar mass: ρ ∝ Σ X_i * M_i
        
        Args:
            pred_log: Predicted state (batch, 11)
            true_log: True state (batch, 11)
        """
        # Convert to linear space
        X_pred = torch.pow(10, pred_log[:, :self.n_species])
        X_true = torch.pow(10, true_log[:, :self.n_species])
        rho_pred = torch.pow(10, pred_log[:, 9])
        rho_true = torch.pow(10, true_log[:, 9])
        
        # Calculate average molar mass for mixture
        molar_mass_tensor = torch.tensor(
            [self.molar_masses[name] for name in self.species_names],
            device=X_pred.device, dtype=X_pred.dtype
        )
        
        avg_M_pred = torch.sum(X_pred * molar_mass_tensor, dim=1)
        avg_M_true = torch.sum(X_true * molar_mass_tensor, dim=1)
        
        # Mass should be proportional: ρ_pred/ρ_true ≈ M_pred/M_true
        ratio_rho = rho_pred / (rho_true + 1e-10)
        ratio_M = avg_M_pred / (avg_M_true + 1e-10)
        
        return torch.mean((ratio_rho - ratio_M) ** 2)
    
    def monotonicity_loss(self, pred_log, true_log, input_log):
        """
        Temperature monotonicity: for exothermic reactions approaching equilibrium,
        temperature should decrease or stay constant (never increase).
        
        This is a soft constraint with small weight.
        
        Args:
            pred_log: Predicted state (batch, 11)
            true_log: True state (batch, 11)
            input_log: Input state (batch, 12)
        """
        T_in = torch.pow(10, input_log[:, 10])   # Current T
        T_out = torch.pow(10, pred_log[:, 10])   # Next T
        
        # Penalize temperature increases (should be rare for equilibration)
        delta_T = T_out - T_in
        violation = torch.relu(delta_T)  # Only penalize increases
        
        return torch.mean(violation ** 2)


# ==========================================
# Training Framework
# ==========================================
class SequentialPINNTrainer:
    """Complete training framework for Sequential PINN with log inputs."""
    
    def __init__(self, config=Config, device='cpu'):
        self.config = config
        self.device = device
        
        # Create model
        self.model = SequentialPINNLog(
            input_dim=config.INPUT_DIM,
            output_dim=config.OUTPUT_DIM,
            hidden_dims=config.HIDDEN_DIMS
        ).to(device)
        
        # Create physics loss calculator
        self.physics_loss = PhysicsLoss(config)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=200
        )
        
        # Loss history
        self.loss_history = {
            'total': [], 'data': [], 'physics': [],
            'atom': [], 'mass': [], 'monotonicity': []
        }
    
    def compute_loss(self, inputs, targets):
        """
        Compute total loss = data loss + physics losses.
        
        Args:
            inputs: Input tensor (batch, 12) in log scale
            targets: Target tensor (batch, 11) in log scale
        
        Returns:
            total_loss, loss_dict
        """
        # Forward pass
        predictions = self.model(inputs)
        
        # Data fitting loss (MSE in log space)
        loss_data = torch.mean((predictions - targets) ** 2)
        
        # Physics losses (computed in appropriate space)
        loss_atom = self.physics_loss.atom_conservation_loss(predictions, targets)
        loss_mass = self.physics_loss.mass_conservation_loss(predictions, targets)
        loss_mono = self.physics_loss.monotonicity_loss(predictions, targets, inputs)
        
        # Total physics loss
        loss_physics = (
            self.config.LAMBDA_ATOM * loss_atom +
            self.config.LAMBDA_MASS * loss_mass +
            self.config.LAMBDA_MONOTONICITY * loss_mono
        )
        
        # Total loss
        total_loss = self.config.LAMBDA_DATA * loss_data + loss_physics
        
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'physics': loss_physics.item(),
            'atom': loss_atom.item(),
            'mass': loss_mass.item(),
            'monotonicity': loss_mono.item()
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {key: 0.0 for key in self.loss_history.keys()}
        n_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward and backward pass
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.compute_loss(inputs, targets)
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key, value in loss_dict.items():
                epoch_losses[key] += value
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_losses = {key: 0.0 for key in self.loss_history.keys()}
        n_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                _, loss_dict = self.compute_loss(inputs, targets)
                
                for key, value in loss_dict.items():
                    val_losses[key] += value
                n_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def train(self, train_loader, val_loader):
        """Complete training loop."""
        print("\n" + "=" * 60)
        print("Training Sequential PINN with Logarithmic Inputs")
        print("=" * 60)
        
        best_val_loss = float('inf')
        start_time = timer.time()
        
        for epoch in range(self.config.EPOCHS):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Record history
            for key in self.loss_history:
                self.loss_history[key].append(train_losses[key])
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_model(os.path.join(self.config.OUTPUT_DIR, 'best_model.pt'))
            
            # Print progress
            if (epoch + 1) % self.config.PRINT_EVERY == 0:
                elapsed = timer.time() - start_time
                print(f"\nEpoch {epoch+1}/{self.config.EPOCHS} ({elapsed:.1f}s)")
                print(f"  Train - Total: {train_losses['total']:.6f}, "
                      f"Data: {train_losses['data']:.6f}, "
                      f"Physics: {train_losses['physics']:.6f}")
                print(f"  Val   - Total: {val_losses['total']:.6f}, "
                      f"Data: {val_losses['data']:.6f}, "
                      f"Physics: {val_losses['physics']:.6f}")
                print(f"    Atom: {val_losses['atom']:.6f}, "
                      f"Mass: {val_losses['mass']:.6f}, "
                      f"Mono: {val_losses['monotonicity']:.6f}")
        
        print(f"\nTraining completed in {timer.time() - start_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.6f}")
    
    def predict_single_step(self, state_log, dt_log):
        """
        Predict next state given current state and time step (all in log scale).
        
        Args:
            state_log: Current state [log10(X), log10(rho), log10(T)] (11,)
            dt_log: Time step log10(dt) (scalar)
        
        Returns:
            next_state_log: Predicted next state (11,)
        """
        self.model.eval()
        with torch.no_grad():
            # Concatenate input
            input_vec = np.concatenate([state_log, [dt_log]])
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(self.device)
            
            # Predict
            output = self.model(input_tensor)
            return output.cpu().numpy()[0]
    
    def rollout(self, initial_state_log, time_steps_log):
        """
        Rollout the model for multiple time steps.
        
        Args:
            initial_state_log: Initial state in log scale (11,)
            time_steps_log: Array of log10(dt) values for each step
        
        Returns:
            trajectory: Array of states at each time step (n_steps+1, 11)
        """
        trajectory = [initial_state_log]
        current_state = initial_state_log
        
        for dt_log in time_steps_log:
            next_state = self.predict_single_step(current_state, dt_log)
            trajectory.append(next_state)
            current_state = next_state
        
        return np.array(trajectory)
    
    def save_model(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint['loss_history']
        print(f"Model loaded from {path}")


# ==========================================
# Data Loading
# ==========================================
def load_data(data_path=Config.DATA_PATH, batch_size=Config.BATCH_SIZE):
    """
    Load and prepare data for training.
    
    Data is already in log scale from cantera_pinn_log.py:
        inputs: [log10(X1-X9), log10(rho), log10(T), log10(dt)]
        outputs: [log10(X1-X9), log10(rho), log10(T)]
    """
    print("\n" + "=" * 60)
    print("Loading Data")
    print("=" * 60)
    
    # Load data
    data = np.load(data_path)
    inputs = data['inputs']    # (N, 12) - already in log scale
    outputs = data['outputs']  # (N, 11) - already in log scale
    
    print(f"Loaded {data_path}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    
    # Data statistics
    print("\nData ranges (log10 scale):")
    print(f"  Input concentrations: [{inputs[:, :9].min():.2f}, {inputs[:, :9].max():.2f}]")
    print(f"  Input rho:  [{inputs[:, 9].min():.2f}, {inputs[:, 9].max():.2f}]")
    print(f"  Input T:    [{inputs[:, 10].min():.2f}, {inputs[:, 10].max():.2f}]")
    print(f"  Input dt:   [{inputs[:, 11].min():.2f}, {inputs[:, 11].max():.2f}]")
    
    # Split into train/val (80/20)
    n_samples = len(inputs)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(inputs[train_idx]),
        torch.FloatTensor(outputs[train_idx])
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(inputs[val_idx]),
        torch.FloatTensor(outputs[val_idx])
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader, data


# ==========================================
# Evaluation and Visualization
# ==========================================
def evaluate_and_plot(trainer, data, output_dir=Config.OUTPUT_DIR):
    """
    Evaluate model and generate comparison plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Evaluation and Visualization")
    print("=" * 60)
    
    inputs = data['inputs']
    outputs_true = data['outputs']
    
    # Get predictions
    trainer.model.eval()
    with torch.no_grad():
        inputs_tensor = torch.FloatTensor(inputs).to(trainer.device)
        outputs_pred = trainer.model(inputs_tensor).cpu().numpy()
    
    # Convert back to linear scale for interpretability
    X_true = 10 ** outputs_true[:, :9]
    X_pred = 10 ** outputs_pred[:, :9]
    T_true = 10 ** outputs_true[:, 10]
    T_pred = 10 ** outputs_pred[:, 10]
    rho_true = 10 ** outputs_true[:, 9]
    rho_pred = 10 ** outputs_pred[:, 9]
    
    # Compute errors
    species_names = Config.SPECIES_NAMES
    
    print("\nPer-Species Errors (log10 scale):")
    for i, name in enumerate(species_names):
        mae = np.mean(np.abs(outputs_pred[:, i] - outputs_true[:, i]))
        print(f"  {name:4s}: MAE = {mae:.4f} decades")
    
    temp_rmse = np.sqrt(np.mean((T_pred - T_true) ** 2))
    temp_rel_err = np.mean(np.abs(T_pred - T_true) / T_true) * 100
    print(f"\nTemperature: RMSE = {temp_rmse:.2f} K, Rel. Error = {temp_rel_err:.2f}%")
    
    # Plot training loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.semilogy(trainer.loss_history['total'], label='Total', linewidth=2)
    ax.semilogy(trainer.loss_history['data'], label='Data', linewidth=2)
    ax.semilogy(trainer.loss_history['physics'], label='Physics', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.semilogy(trainer.loss_history['atom'], label='Atom Conservation', linewidth=2)
    ax.semilogy(trainer.loss_history['mass'], label='Mass Conservation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Physics Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.show()
    
    return {
        'X_pred': X_pred,
        'X_true': X_true,
        'T_pred': T_pred,
        'T_true': T_true,
        'outputs_pred_log': outputs_pred,
        'outputs_true_log': outputs_true
    }


# ==========================================
# Main Entry Point
# ==========================================
def main():
    """Main function to run the complete Sequential PINN pipeline."""
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print("=" * 60)
    print("Sequential PINN with Logarithmic Inputs")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader, data = load_data()
    
    # Create trainer
    trainer = SequentialPINNTrainer(config=Config, device=device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Save model
    model_path = os.path.join(Config.OUTPUT_DIR, Config.MODEL_PATH)
    trainer.save_model(model_path)
    
    # Evaluate and plot
    results = evaluate_and_plot(trainer, data)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    print(f"Plots saved to: {Config.OUTPUT_DIR}/")
    
    return trainer, results


if __name__ == "__main__":
    trainer, results = main()
