#!/usr/bin/env python3
"""
DeepONet Training Pipeline

Training pipeline for learning Krylov complexity dynamics:
- Load pre-trained GNN for code embeddings
- Generate complexity training data
- Train DeepONet with physics-informed loss
- Evaluate on test Hamiltonians
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from holographic_qec.dynamics.krylov import (
    compute_krylov_complexity,
    compute_krylov_basis
)
from holographic_qec.dynamics.hamiltonian import (
    HamiltonianBuilder,
    HamiltonianParams,
    generate_hamiltonian_dataset,
    create_initial_states
)
from holographic_qec.dynamics.deeponet import (
    KrylovDeepONet,
    EnhancedDeepONet,
    PhysicsInformedLoss,
    ComplexityDataset,
    create_deeponet
)
from holographic_qec.gnn.code_generator import HolographicCodeGNN
from holographic_qec.codes.dataset import HolographicCodeDataset, generate_small_code_dataset
from holographic_qec.codes.stabilizer import StabilizerCode, create_codespace_state


class TargetNormalizer:
    """Z-score normalization for Krylov complexity targets."""

    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, values: np.ndarray) -> 'TargetNormalizer':
        """Compute normalization statistics from training data."""
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        if self.std < 1e-8:
            self.std = 1.0  # Prevent division by zero
        self.fitted = True
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Reverse normalization for predictions."""
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        return values * self.std + self.mean

    def state_dict(self) -> Dict[str, float]:
        """Get normalizer state for serialization."""
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state: Dict[str, float]):
        """Load normalizer state from checkpoint."""
        self.mean = state['mean']
        self.std = state['std']
        self.fitted = True


class ComplexityTrainingDataset(Dataset):
    """
    Dataset for training DeepONet on Krylov complexity.

    Generates complexity curves for various (code, Hamiltonian) pairs.
    """

    def __init__(
        self,
        code_embeddings: List[np.ndarray],
        hamiltonians: List[Tuple[np.ndarray, HamiltonianParams]],
        n_time_steps: int = 100,
        t_max: float = 10.0,
        n_initial_states: int = 5,
        normalizer: Optional[TargetNormalizer] = None,
        fit_normalizer: bool = False,
        codes: Optional[List[StabilizerCode]] = None
    ):
        """
        Args:
            code_embeddings: List of code embedding vectors
            hamiltonians: List of (H, params) tuples
            n_time_steps: Number of time points
            t_max: Maximum evolution time
            n_initial_states: Initial states per (code, H) pair
            normalizer: Optional TargetNormalizer for C_K values
            fit_normalizer: If True, fit normalizer on this dataset's data
            codes: Optional list of StabilizerCode objects for code-space initial states
        """
        self.code_embeddings = code_embeddings
        self.hamiltonians = hamiltonians
        self.n_time_steps = n_time_steps
        self.t_max = t_max
        self.n_initial_states = n_initial_states
        self.normalizer = normalizer
        self.codes = codes

        # Generate all data points
        self.data = self._generate_data()

        # Fit normalizer on training data if requested
        if self.normalizer is not None and fit_normalizer and not self.normalizer.fitted:
            all_C = np.array([d['C'][0] for d in self.data])
            self.normalizer.fit(all_C)
            print(f"Fitted normalizer: mean={self.normalizer.mean:.4f}, std={self.normalizer.std:.4f}")

    def _generate_data(self) -> List[Dict]:
        """Generate complexity data for all combinations.

        If self.codes is provided, uses code-space projections as initial states
        and pairs each code only with matching-size Hamiltonians.
        Otherwise falls back to random initial states.
        """
        data = []
        times = np.linspace(0, self.t_max, self.n_time_steps)

        if self.codes is not None:
            # Group Hamiltonians by n_qubits
            hams_by_size: Dict[int, List[Tuple[np.ndarray, HamiltonianParams]]] = {}
            for H, params in self.hamiltonians:
                hams_by_size.setdefault(params.n_qubits, []).append((H, params))

            for code_idx, (code_emb, code) in enumerate(zip(self.code_embeddings, self.codes)):
                n = code.n_physical
                matching_hams = hams_by_size.get(n, [])
                if not matching_hams:
                    continue

                # Compute code-space initial state
                try:
                    psi0 = create_codespace_state(code)
                except (ValueError, Exception) as e:
                    continue

                for H, params in matching_hams:
                    try:
                        result = compute_krylov_complexity(
                            H, psi0, self.t_max, self.n_time_steps
                        )
                        complexity = result.complexity
                    except Exception:
                        continue

                    for t_idx, (t, C) in enumerate(zip(times, complexity)):
                        data.append({
                            'code_embedding': code_emb.astype(np.float32),
                            'hamiltonian_params': params.to_vector(),
                            't': np.array([t], dtype=np.float32),
                            'C': np.array([C], dtype=np.float32)
                        })
        else:
            # Fallback: random initial states (original behavior)
            for code_idx, code_emb in enumerate(self.code_embeddings):
                for H, params in self.hamiltonians:
                    n_qubits = params.n_qubits

                    for state_idx in range(self.n_initial_states):
                        psi0 = create_initial_states(n_qubits, 'random')

                        try:
                            result = compute_krylov_complexity(
                                H, psi0, self.t_max, self.n_time_steps
                            )
                            complexity = result.complexity
                        except Exception:
                            continue

                        for t_idx, (t, C) in enumerate(zip(times, complexity)):
                            data.append({
                                'code_embedding': code_emb.astype(np.float32),
                                'hamiltonian_params': params.to_vector(),
                                't': np.array([t], dtype=np.float32),
                                'C': np.array([C], dtype=np.float32)
                            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        C_value = item['C'].copy()

        # Apply normalization if available and fitted
        if self.normalizer is not None and self.normalizer.fitted:
            C_value = self.normalizer.transform(C_value)

        return {
            'code_embedding': torch.tensor(item['code_embedding']),
            'hamiltonian_params': torch.tensor(item['hamiltonian_params']),
            't': torch.tensor(item['t']),
            'C_true': torch.tensor(C_value, dtype=torch.float32)
        }


def load_gnn_embeddings(
    gnn_checkpoint: str,
    codes: List,
    device: torch.device
) -> List[np.ndarray]:
    """
    Load pre-trained GNN and extract embeddings for codes.

    Args:
        gnn_checkpoint: Path to GNN checkpoint
        codes: List of code samples
        device: Computation device

    Returns:
        List of embedding vectors
    """
    # Load checkpoint (weights_only=False for PyTorch 2.6 compatibility)
    checkpoint = torch.load(gnn_checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get('config', {}).get('gnn', {}).get('model', {})

    # Create model
    model = HolographicCodeGNN(
        node_dim=config.get('node_dim', 4),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 4)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Extract embeddings
    embeddings = []

    with torch.no_grad():
        for code in codes:
            # Get graph representation
            if hasattr(code, 'stabilizer_code'):
                sc = code.stabilizer_code
            else:
                sc = code

            node_feat, edge_idx, hyperedges = sc.to_graph()
            x = torch.tensor(node_feat, dtype=torch.float32).to(device)

            # Get embedding
            emb = model.get_embeddings(x, hyperedges)
            embeddings.append(emb.cpu().numpy().squeeze())

    return embeddings


def generate_dummy_embeddings(n_codes: int, embed_dim: int = 128) -> List[np.ndarray]:
    """Generate dummy embeddings for testing without GNN."""
    return [np.random.randn(embed_dim).astype(np.float32) for _ in range(n_codes)]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: PhysicsInformedLoss,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_mono = 0.0
    n_batches = 0

    for batch in dataloader:
        code_emb = batch['code_embedding'].to(device)
        ham_params = batch['hamiltonian_params'].to(device)
        t = batch['t'].to(device)
        C_true = batch['C_true'].to(device)

        # Forward and loss
        optimizer.zero_grad()
        loss, losses = loss_fn(model, code_emb, ham_params, t, C_true)

        # Backward
        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()
        total_mse += losses['mse'].item()
        total_mono += losses['monotonicity'].item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'monotonicity': total_mono / n_batches
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    normalizer: Optional[TargetNormalizer] = None
) -> Dict[str, float]:
    """Evaluate model with optional denormalization for meaningful metrics."""
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in dataloader:
            code_emb = batch['code_embedding'].to(device)
            ham_params = batch['hamiltonian_params'].to(device)
            t = batch['t'].to(device)
            C_true = batch['C_true'].to(device)

            C_pred = model(code_emb, ham_params, t)

            pred_np = C_pred.cpu().numpy().flatten()
            true_np = C_true.cpu().numpy().flatten()

            # Denormalize for meaningful metrics
            if normalizer is not None and normalizer.fitted:
                pred_np = normalizer.inverse_transform(pred_np)
                true_np = normalizer.inverse_transform(true_np)

            all_preds.extend(pred_np)
            all_true.extend(true_np)

    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    # Compute metrics
    mse = np.mean((all_preds - all_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_true))

    # Relative error (avoid division by zero)
    rel_error = np.mean(np.abs(all_preds - all_true) / np.maximum(all_true, 1e-6)) * 100

    # R^2 score
    ss_res = np.sum((all_true - all_preds) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - (ss_res / max(ss_tot, 1e-10))

    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'rel_error': float(rel_error),
        'r2': float(r2)
    }


def train_deeponet(config: Dict) -> Tuple[nn.Module, Dict]:
    """
    Main training function.

    Args:
        config: Configuration dictionary

    Returns:
        Trained model and results dict
    """
    print("=" * 60)
    print("DeepONet Training Pipeline")
    print("=" * 60)

    # Setup device
    device_str = config.get('hardware', {}).get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # Set seed
    seed = config.get('hardware', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Output directories
    output_dir = Path(config.get('output', {}).get('base_dir', 'results'))
    checkpoint_dir = output_dir / config.get('output', {}).get('checkpoints_dir', 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    print("\n[1/5] Generating training data...")
    deeponet_config = config.get('deeponet', {})
    dynamics_config = deeponet_config.get('dynamics', {})

    # Generate Hamiltonians for n_physical in {4,5,6,7,8}
    ham_config = config.get('hamiltonians', {})
    xxz_deltas = ham_config.get('xxz', {}).get('Delta_values', [0.0, 0.5, 1.0, 2.0])
    ising_hs = ham_config.get('ising', {}).get('h_values', [0.0, 0.5, 1.0, 2.0])
    n_qubits = ham_config.get('xxz', {}).get('n_qubits', [4, 5, 6, 7, 8])

    hamiltonians = generate_hamiltonian_dataset(
        xxz_deltas=xxz_deltas,
        ising_hs=ising_hs,
        n_qubits_list=n_qubits,
        n_random=50
    )
    print(f"Generated {len(hamiltonians)} Hamiltonians")

    # Generate small code dataset (n_physical <= 8)
    print("Generating small code dataset for code-space states...")
    small_code_samples = generate_small_code_dataset(seed=seed)
    all_codes = [sample.code for sample in small_code_samples]

    # Get code embeddings from trained GNN or fall back to dummy
    gnn_checkpoint = config.get('gnn_checkpoint', None)

    if gnn_checkpoint and Path(gnn_checkpoint).exists():
        print("Loading GNN embeddings from trained model...")
        code_embeddings = load_gnn_embeddings(
            gnn_checkpoint,
            all_codes,
            device
        )
        print(f"Extracted {len(code_embeddings)} real GNN embeddings (dim={code_embeddings[0].shape[0]})")
    else:
        print("Using dummy code embeddings (no GNN checkpoint)")
        code_embeddings = generate_dummy_embeddings(
            len(all_codes), deeponet_config.get('model', {}).get('code_dim', 128)
        )

    max_codes = len(code_embeddings)

    # Create dataset
    print("Computing Krylov complexity curves with code-space initial states...")
    t_max = dynamics_config.get('t_max', 10.0)
    n_time_steps = dynamics_config.get('n_time_steps', 100)

    # Stratified Hamiltonian split by n_qubits (80/20 per size)
    rng = np.random.RandomState(seed)
    hams_by_size: Dict[int, List[int]] = {}
    for idx, (H, params) in enumerate(hamiltonians):
        hams_by_size.setdefault(params.n_qubits, []).append(idx)

    train_ham_idx = []
    val_ham_idx = []
    for n_q, indices in sorted(hams_by_size.items()):
        rng.shuffle(indices)
        split = int(0.8 * len(indices))
        train_ham_idx.extend(indices[:split])
        val_ham_idx.extend(indices[split:])

    train_hams = [hamiltonians[i] for i in train_ham_idx]
    val_hams = [hamiltonians[i] for i in val_ham_idx]
    print(f"Hamiltonian split: {len(train_hams)} train / {len(val_hams)} val (stratified by n_qubits)")

    # Create normalizer for Z-score normalization of complexity targets
    target_normalizer = TargetNormalizer()

    train_dataset = ComplexityTrainingDataset(
        code_embeddings[:max_codes],
        train_hams,
        n_time_steps=n_time_steps,
        t_max=t_max,
        n_initial_states=1,
        normalizer=target_normalizer,
        fit_normalizer=True,
        codes=all_codes
    )

    # Validation uses same normalizer (already fitted from training)
    val_dataset = ComplexityTrainingDataset(
        code_embeddings[:max_codes],
        val_hams,
        n_time_steps=n_time_steps,
        t_max=t_max,
        n_initial_states=1,
        normalizer=target_normalizer,
        fit_normalizer=False,
        codes=all_codes
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    training_config = deeponet_config.get('training', {})
    batch_size = training_config.get('batch_size', 64)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    # Create model
    print("\n[2/5] Creating model...")
    model_config = deeponet_config.get('model', {})
    model = create_deeponet(model_config)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.0005),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )

    physics_config = deeponet_config.get('physics', {})
    loss_fn = PhysicsInformedLoss(
        monotonicity_weight=physics_config.get('monotonicity_weight', 0.0),
        growth_rate_weight=physics_config.get('gradient_regularization', 0.0),
        smoothness_weight=physics_config.get('smoothness_weight', 0.0)
    )

    num_epochs = training_config.get('epochs', 100)
    warmup_epochs = training_config.get('warmup_epochs', 10)
    early_stopping_patience = training_config.get('early_stopping_patience', 30)

    # Learning rate scheduler with warmup
    def get_lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)

    # Training loop
    print("\n[3/5] Training...")
    best_val_r2 = float('-inf')
    patience_counter = 0
    best_epoch = 0
    history = {'train_loss': [], 'val_mse': [], 'val_r2': [], 'lr': []}

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device, gradient_clip=1.0
        )

        # Validate with denormalization for meaningful metrics
        val_metrics = evaluate(model, val_loader, device, normalizer=target_normalizer)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Record
        history['train_loss'].append(train_metrics['loss'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(current_lr)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val MSE: {val_metrics['mse']:.4f} | "
                  f"Val R2: {val_metrics['r2']:.4f} | "
                  f"LR: {current_lr:.6f}")

        # Save best based on R² (not MSE)
        if val_metrics['r2'] > best_val_r2:
            best_val_r2 = val_metrics['r2']
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': best_val_r2,
                'val_mse': val_metrics['mse'],
                'normalizer': target_normalizer.state_dict(),  # Save normalizer!
                'config': config
            }, checkpoint_dir / 'best_deeponet.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (best R²: {best_val_r2:.4f} at epoch {best_epoch+1})")
                break

    # Load best (weights_only=False for PyTorch 2.6 compatibility)
    print("\n[4/5] Loading best model...")
    checkpoint = torch.load(checkpoint_dir / 'best_deeponet.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore normalizer from checkpoint
    if 'normalizer' in checkpoint:
        target_normalizer.load_state_dict(checkpoint['normalizer'])
        print(f"Restored normalizer: mean={target_normalizer.mean:.4f}, std={target_normalizer.std:.4f}")

    # Final evaluation with denormalization
    print("\n[5/5] Final evaluation...")
    final_metrics = evaluate(model, val_loader, device, normalizer=target_normalizer)

    print("\nTest Results:")
    print(f"  MSE: {final_metrics['mse']:.6f}")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    print(f"  MAE: {final_metrics['mae']:.4f}")
    print(f"  Relative Error: {final_metrics['rel_error']:.2f}%")
    print(f"  R2 Score: {final_metrics['r2']:.4f}")

    # Save results
    results = {
        'final_metrics': final_metrics,
        'history': history,
        'best_epoch': checkpoint['epoch'],
        'n_params': n_params,
        'normalizer': target_normalizer.state_dict() if target_normalizer.fitted else None
    }

    with open(output_dir / 'deeponet_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)

    return model, results


def main():
    parser = argparse.ArgumentParser(description="Train DeepONet for Krylov complexity")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--gnn-checkpoint', type=str, default=None,
                        help='Path to pre-trained GNN checkpoint')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {config_path}")
        config = {}

    # Apply overrides
    if args.gnn_checkpoint:
        config['gnn_checkpoint'] = args.gnn_checkpoint
    if args.epochs:
        config.setdefault('deeponet', {}).setdefault('training', {})['epochs'] = args.epochs
    if args.device:
        config.setdefault('hardware', {})['device'] = args.device

    # Train
    model, results = train_deeponet(config)


if __name__ == '__main__':
    main()
