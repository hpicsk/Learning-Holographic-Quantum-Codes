#!/usr/bin/env python3
"""
GNN Training Pipeline

Full training pipeline for the HolographicCodeGNN model:
- Dataset generation/loading
- Curriculum learning
- Multi-task training (distance, rate, threshold)
- Evaluation and checkpointing
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
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from holographic_qec.codes.dataset import (
    HolographicCodeDataset,
    CodeSample,
    create_pyg_dataset,
    compute_dataset_statistics
)
from holographic_qec.gnn.code_generator import (
    HolographicCodeGNN,
    MultiTaskLoss,
    create_model
)


class CodeBatchDataset(Dataset):
    """PyTorch Dataset wrapper for code samples."""

    def __init__(self, samples: List[CodeSample]):
        self.samples = samples

        # Precompute graphs
        for sample in self.samples:
            sample.compute_graph()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensors
        x = torch.tensor(sample.node_features, dtype=torch.float32)
        hyperedges = sample.hyperedges

        # Target values
        distance = sample.code.distance if sample.code.distance else 3  # Default
        rate = sample.code.n_logical / max(1, sample.code.n_physical)

        return {
            'x': x,
            'hyperedges': hyperedges,
            'distance': torch.tensor([distance], dtype=torch.float32),
            'rate': torch.tensor([rate], dtype=torch.float32),
            'n_physical': sample.code.n_physical,
            'family': sample.family
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-sized graphs.

    Note: We don't batch graphs together due to hyperedge complexity.
    Instead, we return a list for sequential processing.
    """
    return batch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: MultiTaskLoss,
    device: torch.device,
    gradient_clip: float = 1.0
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_distance_loss = 0.0
    total_rate_loss = 0.0
    n_samples = 0

    for batch in dataloader:
        # Process each sample in batch (not batched due to variable hyperedges)
        batch_loss = 0.0

        for sample in batch:
            x = sample['x'].to(device)
            hyperedges = sample['hyperedges']
            d_true = sample['distance'].to(device)
            r_true = sample['rate'].to(device)

            # Forward pass
            distance, rate, threshold, embeddings = model(x, hyperedges)

            # Compute loss
            loss, losses = loss_fn(
                distance, d_true,
                rate, r_true
            )

            batch_loss += loss
            total_distance_loss += losses['distance'].item()
            total_rate_loss += losses['rate'].item()
            n_samples += 1

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += batch_loss.item()

    return {
        'loss': total_loss / n_samples,
        'distance_loss': total_distance_loss / n_samples,
        'rate_loss': total_rate_loss / n_samples
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: MultiTaskLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation/test set."""
    model.eval()

    total_loss = 0.0
    predictions = {'distance': [], 'rate': []}
    targets = {'distance': [], 'rate': []}

    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                x = sample['x'].to(device)
                hyperedges = sample['hyperedges']
                d_true = sample['distance'].to(device)
                r_true = sample['rate'].to(device)

                # Forward pass
                distance, rate, threshold, embeddings = model(x, hyperedges)

                # Compute loss
                loss, _ = loss_fn(distance, d_true, rate, r_true)
                total_loss += loss.item()

                # Store predictions
                predictions['distance'].append(distance.item())
                predictions['rate'].append(rate.item())
                targets['distance'].append(d_true.item())
                targets['rate'].append(r_true.item())

    n_samples = len(predictions['distance'])

    # Compute metrics
    d_pred = np.array(predictions['distance'])
    d_true = np.array(targets['distance'])
    r_pred = np.array(predictions['rate'])
    r_true = np.array(targets['rate'])

    # Distance metrics
    d_mae = np.mean(np.abs(d_pred - d_true))
    d_rmse = np.sqrt(np.mean((d_pred - d_true) ** 2))
    d_rel_error = np.mean(np.abs(d_pred - d_true) / np.maximum(d_true, 1)) * 100

    # Rate metrics
    r_mae = np.mean(np.abs(r_pred - r_true))
    r_rmse = np.sqrt(np.mean((r_pred - r_true) ** 2))

    return {
        'loss': total_loss / n_samples,
        'distance_mae': float(d_mae),
        'distance_rmse': float(d_rmse),
        'distance_rel_error': float(d_rel_error),
        'rate_mae': float(r_mae),
        'rate_rmse': float(r_rmse)
    }


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Dict,
    num_epochs: int
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    scheduler_type = config.get('scheduler', 'cosine')

    if scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
    elif scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.5
        )
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
    else:
        return None


def train_gnn(config: Dict) -> Tuple[nn.Module, Dict]:
    """
    Main training function.

    Args:
        config: Configuration dictionary

    Returns:
        Trained model and results dict
    """
    print("=" * 60)
    print("GNN Training Pipeline")
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

    # Create output directories
    output_dir = Path(config.get('output', {}).get('base_dir', 'results'))
    checkpoint_dir = output_dir / config.get('output', {}).get('checkpoints_dir', 'checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generate/load dataset
    print("\n[1/5] Preparing dataset...")
    dataset_config = config.get('dataset', {})

    dataset = HolographicCodeDataset(
        n_happy=dataset_config.get('n_happy', 100),
        n_ldpc=dataset_config.get('n_ldpc', 50),
        n_random=dataset_config.get('n_random', 20),
        train_split=dataset_config.get('train_split', 0.7),
        val_split=dataset_config.get('val_split', 0.15),
        seed=dataset_config.get('seed', 42),
        cache_dir=dataset_config.get('cache_dir', None)
    )
    dataset.generate(parallel=False)

    train_samples, val_samples, test_samples = dataset.get_splits()
    print(f"Dataset sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Print statistics
    stats = compute_dataset_statistics(train_samples)
    print(f"Family distribution: {stats['families']}")

    # Create dataloaders
    train_dataset = CodeBatchDataset(train_samples)
    val_dataset = CodeBatchDataset(val_samples)
    test_dataset = CodeBatchDataset(test_samples)

    batch_size = config.get('gnn', {}).get('training', {}).get('batch_size', 32)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # Create model
    print("\n[2/5] Creating model...")
    model_config = config.get('gnn', {}).get('model', {})
    model = create_model(model_config, model_type='predictor')
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Create optimizer and loss
    training_config = config.get('gnn', {}).get('training', {})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )

    num_epochs = training_config.get('epochs', 100)
    scheduler = create_scheduler(optimizer, training_config, num_epochs)
    loss_fn = MultiTaskLoss(num_tasks=3)

    # Training loop
    print("\n[3/5] Training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = training_config.get('early_stopping_patience', 20)
    gradient_clip = training_config.get('gradient_clip', 1.0)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_d_loss': [], 'val_d_mae': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device, gradient_clip
        )

        # Validate
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_d_loss'].append(train_metrics['distance_loss'])
        history['val_d_mae'].append(val_metrics['distance_mae'])
        history['lr'].append(current_lr)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val MAE(d): {val_metrics['distance_mae']:.3f} | "
                  f"LR: {current_lr:.2e}")

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, checkpoint_dir / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model (weights_only=False for PyTorch 2.6 compatibility)
    print("\n[4/5] Loading best model...")
    checkpoint = torch.load(checkpoint_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    print("\n[5/5] Final evaluation...")
    test_metrics = evaluate(model, test_loader, loss_fn, device)

    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Distance MAE: {test_metrics['distance_mae']:.3f}")
    print(f"  Distance RMSE: {test_metrics['distance_rmse']:.3f}")
    print(f"  Distance Relative Error: {test_metrics['distance_rel_error']:.2f}%")
    print(f"  Rate MAE: {test_metrics['rate_mae']:.4f}")

    # Save results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'config': config,
        'best_epoch': checkpoint['epoch'],
        'n_params': n_params
    }

    with open(output_dir / 'gnn_results.json', 'w') as f:
        json.dump({k: v for k, v in results.items() if k != 'config'}, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)

    return model, results


def main():
    parser = argparse.ArgumentParser(description="Train HolographicCodeGNN")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['debug', 'full', 'figures'],
                        help='Use preset configuration')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
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
        print("Using default configuration")
        config = {}

    # Apply preset
    if args.preset and 'presets' in config:
        preset = config['presets'].get(args.preset, {})
        # Deep merge preset into config
        def deep_merge(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        deep_merge(config, preset)

    # Apply command line overrides
    if args.epochs is not None:
        config.setdefault('gnn', {}).setdefault('training', {})['epochs'] = args.epochs
    if args.batch_size is not None:
        config.setdefault('gnn', {}).setdefault('training', {})['batch_size'] = args.batch_size
    if args.device is not None:
        config.setdefault('hardware', {})['device'] = args.device

    # Train
    model, results = train_gnn(config)


if __name__ == '__main__':
    main()
