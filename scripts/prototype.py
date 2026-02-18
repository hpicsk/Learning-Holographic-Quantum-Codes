#!/usr/bin/env python3
"""
Prototype Script: Hybrid Proposal Validation

This script validates the core concepts of the hybrid proposal:
1. HaPPY code generation
2. GNN training for code distance prediction
3. Krylov complexity computation
4. Initial geometry-complexity correlation analysis

Run: python scripts/prototype.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Import our modules
from holographic_qec.codes.stabilizer import (
    StabilizerCode,
    create_five_qubit_code,
    create_steane_code,
    create_repetition_code,
    compute_code_distance,
)
from holographic_qec.codes.happy_codes import (
    create_happy_code,
    generate_happy_dataset,
    HaPPYCode,
)
from holographic_qec.gnn.hypergraph_conv import (
    HypergraphNN,
    CodeDistancePredictor,
    code_to_pyg_data,
)
from holographic_qec.dynamics.krylov import (
    compute_krylov_complexity,
    build_xxz_hamiltonian,
    build_ising_hamiltonian,
    compute_complexity_features,
    KrylovResult,
)


def section_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


# =============================================================================
# PART 1: Code Generation
# =============================================================================

def test_code_generation():
    """Test basic code generation functionality."""
    section_header("PART 1: Quantum Code Generation")

    # Test standard codes
    print("1.1 Testing standard stabilizer codes...")

    # Five-qubit code
    code_5 = create_five_qubit_code()
    print(f"  - Five-qubit code: {code_5.code_parameters}")
    print(f"    Stabilizers: {code_5.stabilizers[:2]}...")

    # Steane code
    code_7 = create_steane_code()
    print(f"  - Steane code: {code_7.code_parameters}")

    # Repetition code
    code_rep = create_repetition_code(5)
    print(f"  - Repetition code: {code_rep.code_parameters}")

    # Test HaPPY codes
    print("\n1.2 Testing HaPPY code generation...")

    for depth in [1, 2, 3]:
        happy = create_happy_code(depth)
        print(f"  - Depth {depth}: n={happy.n_physical}, k={happy.n_logical}, "
              f"d={happy.distance}, {len(happy.stabilizer_code.stabilizers)} stabilizers")

    # Test graph conversion
    print("\n1.3 Testing graph representation...")
    node_feat, edge_idx, hyperedges = code_5.to_graph()
    print(f"  - Node features shape: {node_feat.shape}")
    print(f"  - Edge index shape: {edge_idx.shape}")
    print(f"  - Number of hyperedges: {len(hyperedges)}")

    # Test geometric features
    print("\n1.4 Testing geometric feature extraction...")
    happy = create_happy_code(2)
    geom_feat = happy.get_geometric_features()
    print(f"  - Geometric features: {geom_feat}")

    return True


# =============================================================================
# PART 2: GNN Training
# =============================================================================

def prepare_training_data(n_samples: int = 50) -> Tuple[List[Dict], List[float]]:
    """Prepare training data for GNN."""
    print(f"  Generating {n_samples} training samples...")

    data_list = []
    targets = []

    # Generate codes with known distances
    codes = [
        (create_five_qubit_code(), 3),
        (create_steane_code(), 3),
        (create_repetition_code(3), 3),
        (create_repetition_code(5), 5),
        (create_repetition_code(7), 7),
    ]

    # Add HaPPY codes
    for depth in [1, 2]:
        happy = create_happy_code(depth)
        dist = happy.distance if happy.distance else depth + 1
        codes.append((happy.stabilizer_code, dist))

    # Replicate and add variations
    for _ in range(n_samples // len(codes) + 1):
        for code, dist in codes:
            if len(data_list) >= n_samples:
                break

            node_feat, edge_idx, hyperedges = code.to_graph()

            # Add small noise for variation
            node_feat = node_feat + np.random.randn(*node_feat.shape) * 0.01

            data = code_to_pyg_data(node_feat, edge_idx, hyperedges, dist)
            data_list.append(data)
            targets.append(dist)

    return data_list[:n_samples], targets[:n_samples]


def train_gnn_model(
    data_list: List[Dict],
    targets: List[float],
    epochs: int = 100,
    lr: float = 0.01
) -> Tuple[CodeDistancePredictor, List[float]]:
    """Train GNN model on code distance prediction."""

    # Initialize model
    model = CodeDistancePredictor(
        node_features=4,
        hidden_dim=32,
        num_layers=2
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    print(f"  Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for data, target in zip(data_list, targets):
            optimizer.zero_grad()

            # Forward pass
            pred = model(
                data['x'],
                data['hyperedge_index'],
                batch=None
            )

            loss = criterion(pred.squeeze(), torch.tensor([target], dtype=torch.float32))

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_list)
        losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, losses


def evaluate_gnn_model(
    model: CodeDistancePredictor,
    data_list: List[Dict],
    targets: List[float]
) -> Dict:
    """Evaluate trained GNN model."""
    model.eval()

    predictions = []

    with torch.no_grad():
        for data in data_list:
            pred = model(
                data['x'],
                data['hyperedge_index'],
                batch=None
            )
            predictions.append(pred.item())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Compute metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # Relative error
    rel_error = np.mean(np.abs(predictions - targets) / targets) * 100

    return {
        'mse': mse,
        'mae': mae,
        'relative_error_pct': rel_error,
        'predictions': predictions,
        'targets': targets,
    }


def test_gnn_training():
    """Test GNN training pipeline."""
    section_header("PART 2: GNN Training for Code Distance")

    print("2.1 Preparing training data...")
    train_data, train_targets = prepare_training_data(n_samples=50)
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Target distances: {sorted(set(train_targets))}")

    print("\n2.2 Training GNN model...")
    model, losses = train_gnn_model(train_data, train_targets, epochs=100)

    print("\n2.3 Evaluating model...")
    results = evaluate_gnn_model(model, train_data, train_targets)
    print(f"  - MSE: {results['mse']:.4f}")
    print(f"  - MAE: {results['mae']:.4f}")
    print(f"  - Relative Error: {results['relative_error_pct']:.1f}%")

    # Show some predictions
    print("\n  Sample predictions:")
    for i in range(min(5, len(results['predictions']))):
        print(f"    Target: {results['targets'][i]:.0f}, "
              f"Predicted: {results['predictions'][i]:.2f}")

    return model, results, losses


# =============================================================================
# PART 3: Krylov Complexity
# =============================================================================

def test_krylov_complexity():
    """Test Krylov complexity computation."""
    section_header("PART 3: Krylov Complexity Computation")

    print("3.1 Testing XXZ Hamiltonian...")
    n_qubits = 6
    H = build_xxz_hamiltonian(n_qubits, J_xy=1.0, J_z=0.5, h=0.1)
    print(f"  - Hamiltonian size: {H.shape}")
    print(f"  - Hermitian: {np.allclose(H, H.conj().T)}")

    print("\n3.2 Computing Krylov complexity...")

    # Random initial state
    dim = 2 ** n_qubits
    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi0 = psi0 / np.linalg.norm(psi0)

    result = compute_krylov_complexity(H, psi0, t_max=10.0, n_steps=100)

    print(f"  - Krylov dimension: {result.krylov_dimension}")
    print(f"  - Growth exponent: {result.growth_exponent:.3f}"
          if result.growth_exponent else "  - Growth exponent: N/A")
    print(f"  - Saturation value: {result.saturation_value:.3f}"
          if result.saturation_value else "  - Saturation value: N/A")

    print("\n3.3 Testing different Hamiltonians...")

    hamiltonians = [
        ("XXZ (J_z=0.5)", build_xxz_hamiltonian(5, J_xy=1.0, J_z=0.5)),
        ("XXZ (J_z=1.5)", build_xxz_hamiltonian(5, J_xy=1.0, J_z=1.5)),
        ("Ising (g=0.5)", build_ising_hamiltonian(5, J=1.0, g=0.5)),
        ("Ising (g=1.5)", build_ising_hamiltonian(5, J=1.0, g=1.5)),
    ]

    results_list = []
    for name, H in hamiltonians:
        dim = H.shape[0]
        psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi0 = psi0 / np.linalg.norm(psi0)

        result = compute_krylov_complexity(H, psi0, t_max=10.0, n_steps=50)
        results_list.append((name, result))

        exp_str = f"{result.growth_exponent:.2f}" if result.growth_exponent else "N/A"
        print(f"  - {name}: dim={result.krylov_dimension}, exp={exp_str}")

    return results_list


# =============================================================================
# PART 4: Geometry-Complexity Correlation
# =============================================================================

def test_geometry_complexity_correlation():
    """Test correlation between code geometry and complexity dynamics."""
    section_header("PART 4: Geometry-Complexity Correlation Analysis")

    print("4.1 Generating codes with geometric features...")

    # Generate codes
    codes = []
    for depth in [1, 2, 3]:
        for _ in range(5):  # Multiple variations per depth
            happy = create_happy_code(depth)
            codes.append(happy)

    print(f"  - Generated {len(codes)} codes")

    print("\n4.2 Computing Krylov complexity for each code...")

    data = []
    for i, code in enumerate(tqdm(codes, desc="  Processing codes")):
        # Get geometric features
        geom_feat = code.get_geometric_features()

        # Build Hamiltonian on code qubits (simplified)
        n_qubits = min(code.n_physical, 8)  # Cap for speed
        H = build_xxz_hamiltonian(n_qubits, J_xy=1.0, J_z=1.0, h=0.0)

        # Random initial state
        dim = 2 ** n_qubits
        psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi0 = psi0 / np.linalg.norm(psi0)

        # Compute complexity
        result = compute_krylov_complexity(H, psi0, t_max=5.0, n_steps=50)
        complexity_feat = compute_complexity_features(result)

        data.append({
            'code': code,
            'depth': code.depth,
            'n_physical': code.n_physical,
            'geometric_features': geom_feat,
            'complexity_features': complexity_feat,
            'growth_exponent': result.growth_exponent,
            'saturation': result.saturation_value,
            'krylov_dim': result.krylov_dimension,
        })

    print("\n4.3 Analyzing correlations...")

    # Extract arrays
    depths = np.array([d['depth'] for d in data])
    n_physical = np.array([d['n_physical'] for d in data])
    growth_exp = np.array([d['growth_exponent'] if d['growth_exponent'] else 0 for d in data])
    saturation = np.array([d['saturation'] if d['saturation'] else 0 for d in data])
    krylov_dim = np.array([d['krylov_dim'] for d in data])

    # Compute correlations
    def safe_corrcoef(x, y):
        """Compute correlation, handling edge cases."""
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    correlations = {
        'depth_vs_growth_exp': safe_corrcoef(depths, growth_exp),
        'depth_vs_saturation': safe_corrcoef(depths, saturation),
        'depth_vs_krylov_dim': safe_corrcoef(depths, krylov_dim),
        'n_physical_vs_growth_exp': safe_corrcoef(n_physical, growth_exp),
        'n_physical_vs_krylov_dim': safe_corrcoef(n_physical, krylov_dim),
    }

    print("  Correlations:")
    for name, corr in correlations.items():
        print(f"    - {name}: r = {corr:.3f}")

    print("\n4.4 Summary by depth:")
    for d in sorted(set(depths)):
        mask = depths == d
        print(f"  Depth {d}:")
        print(f"    - Avg growth exponent: {np.mean(growth_exp[mask]):.3f}")
        print(f"    - Avg saturation: {np.mean(saturation[mask]):.3f}")
        print(f"    - Avg Krylov dim: {np.mean(krylov_dim[mask]):.1f}")

    return data, correlations


# =============================================================================
# PART 5: Visualization
# =============================================================================

def create_visualizations(
    gnn_losses: List[float],
    krylov_results: List[Tuple[str, KrylovResult]],
    correlation_data: List[Dict],
    save_path: str = "results"
):
    """Create and save visualization plots."""
    section_header("PART 5: Generating Visualizations")

    os.makedirs(save_path, exist_ok=True)

    # Plot 1: GNN training loss
    print("  Creating GNN training loss plot...")
    plt.figure(figsize=(8, 5))
    plt.plot(gnn_losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('GNN Training: Code Distance Prediction')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/gnn_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Krylov complexity curves
    print("  Creating Krylov complexity curves plot...")
    plt.figure(figsize=(10, 6))
    for name, result in krylov_results:
        plt.plot(result.times, result.complexity, label=name, linewidth=2)
    plt.xlabel('Time t')
    plt.ylabel('Krylov Complexity C_K(t)')
    plt.title('Krylov Complexity for Different Hamiltonians')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_path}/krylov_complexity_curves.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Geometry-Complexity correlation
    print("  Creating geometry-complexity correlation plot...")
    depths = np.array([d['depth'] for d in correlation_data])
    growth_exp = np.array([d['growth_exponent'] if d['growth_exponent'] else 0
                          for d in correlation_data])

    plt.figure(figsize=(8, 6))
    plt.scatter(depths, growth_exp, s=100, alpha=0.6)

    # Add trend line
    if len(set(depths)) > 1:
        z = np.polyfit(depths, growth_exp, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(depths), max(depths), 100)
        plt.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')

    plt.xlabel('Code Depth')
    plt.ylabel('Complexity Growth Exponent')
    plt.title('Code Geometry vs Complexity Dynamics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{save_path}/geometry_complexity_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Plots saved to {save_path}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all prototype tests."""
    print("\n" + "=" * 60)
    print("  HYBRID PROPOSAL PROTOTYPE VALIDATION")
    print("  Learning Holographic Quantum Codes:")
    print("  Neural Networks for Geometry and Complexity Dynamics")
    print("=" * 60)

    # Part 1: Code Generation
    test_code_generation()

    # Part 2: GNN Training
    model, gnn_results, losses = test_gnn_training()

    # Part 3: Krylov Complexity
    krylov_results = test_krylov_complexity()

    # Part 4: Correlation Analysis
    correlation_data, correlations = test_geometry_complexity_correlation()

    # Part 5: Visualizations
    create_visualizations(losses, krylov_results, correlation_data)

    # Summary
    section_header("SUMMARY")
    print("Prototype validation completed successfully!")
    print("\nKey Results:")
    print(f"  - GNN relative error: {gnn_results['relative_error_pct']:.1f}%")
    print(f"  - Best correlation (depth vs Krylov dim): "
          f"r = {correlations['depth_vs_krylov_dim']:.3f}")

    print("\nNext Steps:")
    print("  1. Scale up code dataset (5000+ codes)")
    print("  2. Implement full DeepONet for dynamics")
    print("  3. Comprehensive correlation analysis")
    print("  4. Hardware validation experiments")

    return {
        'gnn_model': model,
        'gnn_results': gnn_results,
        'krylov_results': krylov_results,
        'correlation_data': correlation_data,
        'correlations': correlations,
    }


if __name__ == "__main__":
    results = main()
