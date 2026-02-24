"""
Physics-Informed Neural Network for Chemical Kinetics
Built from scratch with optimal scaling and physics constraints

Architecture:
    Input:  [log(X₁), ..., log(X₉), log(ρ/ρ₀), log(T/T₀), log(dt)] → 12 features
    Output: [log(X₁'), ..., log(X₉'), log(ρ'/ρ₀), log(T'/T₀)]      → 11 features

Key Features:
    - All inputs/outputs in log scale for equal importance across magnitudes
    - Reference normalization: ρ₀ = 0.0013 kg/m³, T₀ = 7500 K
    - Physics constraints: atom conservation, energy conservation, mass conservation
    - Loss = λ_data * L_data + λ_atom * L_atom + λ_energy * L_energy + λ_mass * L_mass

Governing Equations:
    1. Atom Conservation: Σ n_i·X_i = constant for each atom type (C, O, N, Ar)
    2. Mass Conservation: ρ·V = constant (constant volume reactor)
    3. Energy Conservation: U = constant (adiabatic, constant volume)

Usage:
    python pinn_from_scratch.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple
import time as timer

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    """Configuration for PINN model and training."""
    
    # ===== REFERENCE VALUES FOR NORMALIZATION =====
    RHO_0 = 0.0013  # kg/m³ - Reference density
    T_0 = 7500.0    # K - Reference temperature
    
    # ===== MODEL ARCHITECTURE =====
    INPUT_DIM = 12   # [log(X₁-₉), log(ρ/ρ₀), log(T/T₀), log(dt)]
    OUTPUT_DIM = 11  # [log(X₁-₉), log(ρ/ρ₀), log(T/T₀)]
    HIDDEN_LAYERS = [512, 512, 256, 256]  # Deep network for complex dynamics
    ACTIVATION = 'tanh'  # 'tanh', 'relu', 'silu'
    
    # ===== TRAINING PARAMETERS =====
    EPOCHS = 1000
    BATCH_SIZE = 1024
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    SCHEDULER_PATIENCE = 300
    SCHEDULER_FACTOR = 0.5
    
    # ===== LOSS WEIGHTS =====
    LAMBDA_DATA = 1.0        # Data fitting
    LAMBDA_ATOM = 10.0       # Atom conservation (critical!)
    LAMBDA_MASS = 5.0        # Mass conservation
    LAMBDA_ENERGY = 5.0      # Energy conservation (constant volume)
    LAMBDA_PHYSICS_RAMP = True  # Gradually increase physics loss weight
    
    # ===== SPECIES AND CHEMISTRY =====
    SPECIES_NAMES = ['CO2', 'O2', 'N2', 'CO', 'NO', 'C', 'O', 'N', 'AR']
    N_SPECIES = 9
    
    # Molar masses (kg/mol)
    MOLAR_MASSES = {
        'CO2': 0.04401,  'O2': 0.032,     'N2': 0.028014,
        'CO': 0.02801,   'NO': 0.03001,   'C': 0.01201,
        'O': 0.016,      'N': 0.014007,   'AR': 0.039948
    }
    
    # Atomic composition matrix: atoms per molecule
    # Format: {species: {atom: count}}
    ATOMIC_COMPOSITION = {
        'CO2': {'C': 1, 'O': 2, 'N': 0, 'AR': 0},
        'O2':  {'C': 0, 'O': 2, 'N': 0, 'AR': 0},
        'N2':  {'C': 0, 'O': 0, 'N': 2, 'AR': 0},
        'CO':  {'C': 1, 'O': 1, 'N': 0, 'AR': 0},
        'NO':  {'C': 0, 'O': 1, 'N': 1, 'AR': 0},
        'C':   {'C': 1, 'O': 0, 'N': 0, 'AR': 0},
        'O':   {'C': 0, 'O': 1, 'N': 0, 'AR': 0},
        'N':   {'C': 0, 'O': 0, 'N': 1, 'AR': 0},
        'AR':  {'C': 0, 'O': 0, 'N': 0, 'AR': 1}
    }
    
    # Gas constant
    R_UNIVERSAL = 8.314  # J/mol/K
    
    # ===== PATHS =====
    DATA_PATH = '/Users/xiaoxizhou/Downloads/su_26/adrian_surf/reaction_data_log.npz'
    MODEL_SAVE_DIR = 'pinn_models'
    OUTPUT_DIR = 'pinn_outputs'
    
    # ===== LOGGING =====
    PRINT_EVERY = 100
    SAVE_CHECKPOINT_EVERY = 500


# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class ChemPINN(nn.Module):
    """
    Physics-Informed Neural Network for Chemical Kinetics.
    
    Maps: (current_state_log, log_dt) → next_state_log
    All quantities in log scale for numerical stability.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Build network
        layers = []
        dims = [config.INPUT_DIM] + config.HIDDEN_LAYERS + [config.OUTPUT_DIM]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            # Activation (not on output layer)
            if i < len(dims) - 2:
                if config.ACTIVATION == 'tanh':
                    layers.append(nn.Tanh())
                elif config.ACTIVATION == 'relu':
                    layers.append(nn.ReLU())
                elif config.ACTIVATION == 'silu':
                    layers.append(nn.SiLU())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, 12) - [log(X₁-₉), log(ρ/ρ₀), log(T/T₀), log(dt)]
        
        Returns:
            (batch, 11) - [log(X₁-₉'), log(ρ'/ρ₀), log(T'/T₀)]
        """
        return self.network(x)


# ==========================================
# PHYSICS-INFORMED LOSS FUNCTIONS
# ==========================================
class PhysicsLoss:
    """
    Implements physics-based constraints as loss functions.
    
    Governing Equations:
        1. Atom Conservation: Σ nᵢ·Xᵢ = constant for each atom type
        2. Mass Conservation: ρ·V = constant (constant volume)
        3. Energy Conservation: U = constant (adiabatic, constant volume)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.species_names = config.SPECIES_NAMES
        self.atomic_comp = config.ATOMIC_COMPOSITION
        self.molar_masses = config.MOLAR_MASSES
        
        # Build atomic composition matrix (4 atoms × 9 species)
        self.atom_types = ['C', 'O', 'N', 'AR']
        self.atom_matrix = self._build_atom_matrix()
    
    def _build_atom_matrix(self) -> torch.Tensor:
        """Build (4, 9) matrix of atomic composition."""
        matrix = []
        for atom in self.atom_types:
            row = [self.atomic_comp[sp][atom] for sp in self.species_names]
            matrix.append(row)
        return torch.tensor(matrix, dtype=torch.float32)
    
    def atom_conservation_loss(self, pred_log: torch.Tensor, 
                               true_log: torch.Tensor,
                               device: torch.device) -> torch.Tensor:
        """
        Atom conservation: Number of each atom type must be constant.
        
        For each atom type k:
            Σᵢ nₖᵢ · Xᵢ = constant
        
        where nₖᵢ is the number of atoms of type k in species i.
        
        Args:
            pred_log: (batch, 11) predictions in log scale
            true_log: (batch, 11) ground truth in log scale
        
        Returns:
            Atom conservation loss
        """
        # Convert to linear space for atom counting
        X_pred = torch.pow(10, pred_log[:, :9])  # (batch, 9)
        X_true = torch.pow(10, true_log[:, :9])  # (batch, 9)
        
        # Move atom matrix to device
        atom_matrix = self.atom_matrix.to(device)  # (4, 9)
        
        # Compute atom counts: (batch, 4) = (batch, 9) @ (9, 4)
        atom_count_pred = X_pred @ atom_matrix.T  # (batch, 4)
        atom_count_true = X_true @ atom_matrix.T  # (batch, 4)
        
        # Relative error in atom counts (should be zero)
        rel_error = (atom_count_pred - atom_count_true) / (atom_count_true + 1e-10)
        loss = torch.mean(rel_error ** 2)
        
        return loss
    
    def mass_conservation_loss(self, pred_log: torch.Tensor,
                               true_log: torch.Tensor,
                               device: torch.device) -> torch.Tensor:
        """
        Mass conservation: ρ·V = constant.
        
        For constant volume: ρ must remain constant.
        But we also check that average molar mass is consistent:
            ρ ∝ Σ Xᵢ · Mᵢ
        
        Args:
            pred_log: (batch, 11) predictions
            true_log: (batch, 11) ground truth
        
        Returns:
            Mass conservation loss
        """
        # Extract density (already normalized by ρ₀)
        log_rho_pred = pred_log[:, 9]  # log(ρ'/ρ₀)
        log_rho_true = true_log[:, 9]  # log(ρ/ρ₀)
        
        # For constant volume, density should not change much
        # (small changes due to temperature effects are OK)
        loss_rho_direct = torch.mean((log_rho_pred - log_rho_true) ** 2)
        
        # Also check consistency with molar mass
        X_pred = torch.pow(10, pred_log[:, :9])
        X_true = torch.pow(10, true_log[:, :9])
        
        # Average molar mass
        molar_mass_vec = torch.tensor(
            [self.molar_masses[sp] for sp in self.species_names],
            device=device, dtype=torch.float32
        )
        
        avg_M_pred = torch.sum(X_pred * molar_mass_vec, dim=1)
        avg_M_true = torch.sum(X_true * molar_mass_vec, dim=1)
        
        # ρ should scale proportionally with average molar mass
        rho_pred_linear = torch.pow(10, log_rho_pred)
        rho_true_linear = torch.pow(10, log_rho_true)
        
        ratio_rho = rho_pred_linear / (rho_true_linear + 1e-10)
        ratio_M = avg_M_pred / (avg_M_true + 1e-10)
        
        loss_consistency = torch.mean((ratio_rho - ratio_M) ** 2)
        
        return loss_rho_direct + loss_consistency
    
    def energy_conservation_loss(self, pred_log: torch.Tensor,
                                 true_log: torch.Tensor) -> torch.Tensor:
        """
        Energy conservation: For adiabatic constant-volume process, U = constant.
        
        This is approximated by checking that ρ·T remains relatively constant
        (since U ∝ ρ·cᵥ·T for ideal gas).
        
        Args:
            pred_log: (batch, 11) predictions
            true_log: (batch, 11) ground truth
        
        Returns:
            Energy conservation loss
        """
        # Extract temperature (normalized by T₀)
        log_T_pred = pred_log[:, 10]  # log(T'/T₀)
        log_T_true = true_log[:, 10]  # log(T/T₀)
        
        # Extract density
        log_rho_pred = pred_log[:, 9]
        log_rho_true = true_log[:, 9]
        
        # Energy proxy: log(ρ·T) = log(ρ) + log(T)
        log_energy_pred = log_rho_pred + log_T_pred
        log_energy_true = log_rho_true + log_T_true
        
        # Should be constant (in log space)
        loss = torch.mean((log_energy_pred - log_energy_true) ** 2)
        
        return loss


# ==========================================
# TRAINING FRAMEWORK
# ==========================================
class PINNTrainer:
    """Complete training framework for chemical kinetics PINN."""
    
    def __init__(self, config: Config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        
        # Create model
        self.model = ChemPINN(config).to(self.device)
        
        # Create physics loss calculator
        self.physics_loss = PhysicsLoss(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE
        )
        
        # Loss history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'data_loss': [], 'atom_loss': [],
            'mass_loss': [], 'energy_loss': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def compute_loss(self, inputs: torch.Tensor, targets: torch.Tensor,
                     epoch: int = 0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss = data loss + physics losses.
        
        Args:
            inputs: (batch, 12) input features
            targets: (batch, 11) target outputs
            epoch: Current epoch (for physics loss ramp-up)
        
        Returns:
            total_loss, loss_dict
        """
        # Forward pass
        predictions = self.model(inputs)
        
        # Data loss (MSE in log space)
        loss_data = torch.mean((predictions - targets) ** 2)
        
        # Physics losses
        loss_atom = self.physics_loss.atom_conservation_loss(
            predictions, targets, self.device
        )
        loss_mass = self.physics_loss.mass_conservation_loss(
            predictions, targets, self.device
        )
        loss_energy = self.physics_loss.energy_conservation_loss(
            predictions, targets
        )
        
        # Physics loss ramp-up (gradually increase weight)
        if self.config.LAMBDA_PHYSICS_RAMP:
            ramp = min(1.0, epoch / 1000.0)  # Reach full weight at epoch 1000
        else:
            ramp = 1.0
        
        # Total loss
        total_loss = (
            self.config.LAMBDA_DATA * loss_data +
            ramp * self.config.LAMBDA_ATOM * loss_atom +
            ramp * self.config.LAMBDA_MASS * loss_mass +
            ramp * self.config.LAMBDA_ENERGY * loss_energy
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'data': loss_data.item(),
            'atom': loss_atom.item(),
            'mass': loss_mass.item(),
            'energy': loss_energy.item(),
            'ramp': ramp
        }
        
        return total_loss, loss_dict
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'data': 0, 'atom': 0, 'mass': 0, 'energy': 0}
        n_batches = 0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Compute loss and gradients
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.compute_loss(inputs, targets, epoch)
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate the model."""
        self.model.eval()
        val_losses = {'total': 0, 'data': 0, 'atom': 0, 'mass': 0, 'energy': 0}
        n_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                _, loss_dict = self.compute_loss(inputs, targets, epoch)
                
                for key in val_losses:
                    val_losses[key] += loss_dict[key]
                n_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= n_batches
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Complete training loop."""
        print("\n" + "=" * 80)
        print("TRAINING PHYSICS-INFORMED NEURAL NETWORK")
        print("=" * 80)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print("\nLoss weights:")
        print(f"  Data: {self.config.LAMBDA_DATA}")
        print(f"  Atom conservation: {self.config.LAMBDA_ATOM}")
        print(f"  Mass conservation: {self.config.LAMBDA_MASS}")
        print(f"  Energy conservation: {self.config.LAMBDA_ENERGY}")
        
        start_time = timer.time()
        
        for epoch in range(self.config.EPOCHS):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_losses = self.validate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Record history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['data_loss'].append(val_losses['data'])
            self.history['atom_loss'].append(val_losses['atom'])
            self.history['mass_loss'].append(val_losses['mass'])
            self.history['energy_loss'].append(val_losses['energy'])
            
            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.save_model('best_model.pt')
            
            # Print progress
            if (epoch + 1) % self.config.PRINT_EVERY == 0:
                elapsed = timer.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch {epoch+1}/{self.config.EPOCHS} ({elapsed:.1f}s, lr={lr:.2e})")
                print(f"  Train Loss: {train_losses['total']:.6f}")
                print(f"  Val Loss:   {val_losses['total']:.6f}")
                print(f"    Data: {val_losses['data']:.6f}, "
                      f"Atom: {val_losses['atom']:.6f}, "
                      f"Mass: {val_losses['mass']:.6f}, "
                      f"Energy: {val_losses['energy']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pt')
        
        total_time = timer.time() - start_time
        print(f"\n" + "=" * 80)
        print(f"TRAINING COMPLETE")
        print(f"=" * 80)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")
    
    def predict_single_step(self, state_log: np.ndarray, 
                           log_dt: float) -> np.ndarray:
        """
        Predict next state from current state and time step.
        
        Args:
            state_log: (11,) current state in log scale
            log_dt: log10(dt)
        
        Returns:
            next_state_log: (11,) predicted next state
        """
        self.model.eval()
        with torch.no_grad():
            # Prepare input
            input_vec = np.concatenate([state_log, [log_dt]])
            input_tensor = torch.FloatTensor(input_vec).unsqueeze(0).to(self.device)
            
            # Predict
            output = self.model(input_tensor)
            return output.cpu().numpy()[0]
    
    def save_model(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.MODEL_SAVE_DIR, exist_ok=True)
        path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config
        }, path)
    
    def load_model(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.config.MODEL_SAVE_DIR, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_epoch = checkpoint['best_epoch']
        
        print(f"Model loaded from {path}")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch+1}")


# ==========================================
# DATA LOADING AND PREPROCESSING
# ==========================================
def load_and_prepare_data(config: Config) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Load data and create train/val loaders.
    
    Expected data format (from cantera_pinn_log.py):
        inputs: (N, 12) - [log(X₁-₉), log(ρ), log(T), log(dt)]
        outputs: (N, 11) - [log(X₁-₉), log(ρ), log(T)]
    
    We need to normalize ρ and T by reference values.
    """
    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATA")
    print("=" * 80)
    
    # Load data
    data = np.load(config.DATA_PATH)
    inputs = data['inputs'].astype(np.float32)    # (N, 12)
    outputs = data['outputs'].astype(np.float32)  # (N, 11)
    
    print(f"Loaded {config.DATA_PATH}")
    print(f"  Total samples: {len(inputs)}")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Output shape: {outputs.shape}")
    
    # Normalize density and temperature by reference values
    # Input indices: [0-8: species, 9: log(ρ), 10: log(T), 11: log(dt)]
    # Output indices: [0-8: species, 9: log(ρ), 10: log(T)]
    
    print(f"\nNormalizing by reference values:")
    print(f"  ρ₀ = {config.RHO_0} kg/m³")
    print(f"  T₀ = {config.T_0} K")
    
    # Convert log(ρ) to log(ρ/ρ₀) = log(ρ) - log(ρ₀)
    inputs[:, 9] = inputs[:, 9] - np.log10(config.RHO_0)
    outputs[:, 9] = outputs[:, 9] - np.log10(config.RHO_0)
    
    # Convert log(T) to log(T/T₀) = log(T) - log(T₀)
    inputs[:, 10] = inputs[:, 10] - np.log10(config.T_0)
    outputs[:, 10] = outputs[:, 10] - np.log10(config.T_0)
    
    # Print normalized ranges
    print("\nNormalized data ranges:")
    print(f"  log(ρ/ρ₀): [{inputs[:, 9].min():.3f}, {inputs[:, 9].max():.3f}]")
    print(f"  log(T/T₀):  [{inputs[:, 10].min():.3f}, {inputs[:, 10].max():.3f}]")
    print(f"  log(dt):    [{inputs[:, 11].min():.3f}, {inputs[:, 11].max():.3f}]")
    
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(train_dataset)} samples ({len(train_dataset)/n_samples*100:.1f}%)")
    print(f"  Validation: {len(val_dataset)} samples ({len(val_dataset)/n_samples*100:.1f}%)")
    
    # Return raw data for later use
    raw_data = {
        'inputs': inputs,
        'outputs': outputs,
        'train_idx': train_idx,
        'val_idx': val_idx
    }
    
    return train_loader, val_loader, raw_data


# ==========================================
# VISUALIZATION
# ==========================================
def plot_training_history(trainer: PINNTrainer, output_dir: str):
    """Plot training and validation losses."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.semilogy(trainer.history['train_loss'], label='Train', linewidth=2)
    ax.semilogy(trainer.history['val_loss'], label='Validation', linewidth=2)
    ax.axvline(trainer.best_epoch, color='r', linestyle='--', alpha=0.5, 
              label=f'Best (epoch {trainer.best_epoch+1})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Data loss
    ax = axes[0, 1]
    ax.semilogy(trainer.history['data_loss'], label='Data Loss', linewidth=2, color='blue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Data Fitting Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Physics losses
    ax = axes[1, 0]
    ax.semilogy(trainer.history['atom_loss'], label='Atom Conservation', linewidth=2)
    ax.semilogy(trainer.history['mass_loss'], label='Mass Conservation', linewidth=2)
    ax.semilogy(trainer.history['energy_loss'], label='Energy Conservation', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Physics Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Loss components (stacked area)
    ax = axes[1, 1]
    epochs = np.arange(len(trainer.history['data_loss']))
    ax.plot(epochs, trainer.history['data_loss'], label='Data', linewidth=2)
    ax.plot(epochs, trainer.history['atom_loss'], label='Atom', linewidth=2)
    ax.plot(epochs, trainer.history['mass_loss'], label='Mass', linewidth=2)
    ax.plot(epochs, trainer.history['energy_loss'], label='Energy', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (linear scale)')
    ax.set_title('Loss Components (Linear Scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved training history: {path}")
    plt.show()


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    """Main training pipeline."""
    
    # Setup
    config = Config()
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    print("=" * 80)
    print("PHYSICS-INFORMED NEURAL NETWORK FOR CHEMICAL KINETICS")
    print("=" * 80)
    print(f"Device: {device}")
    
    # Create output directories
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Load data
    train_loader, val_loader, raw_data = load_and_prepare_data(config)
    
    # Create trainer
    trainer = PINNTrainer(config, device=device)
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Plot training history
    plot_training_history(trainer, config.OUTPUT_DIR)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Best model saved: {config.MODEL_SAVE_DIR}/best_model.pt")
    print(f"Outputs saved: {config.OUTPUT_DIR}/")
    
    return trainer, raw_data


if __name__ == "__main__":
    trainer, data = main()
