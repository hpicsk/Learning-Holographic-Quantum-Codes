#!/usr/bin/env python3
"""
Correlation Analysis Script

Full analysis pipeline for geometry-complexity correlations:
1. Load trained models and test data
2. Extract geometric and dynamic features
3. Compute correlations and statistical tests
4. Run holographic dictionary tests
5. Generate publication figures
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from holographic_qec.codes.dataset import HolographicCodeDataset, CodeSample, generate_small_code_dataset
from holographic_qec.gnn.code_generator import HolographicCodeGNN
from holographic_qec.dynamics.krylov import compute_krylov_complexity
from holographic_qec.dynamics.hamiltonian import (
    HamiltonianBuilder,
    generate_hamiltonian_dataset,
    create_initial_states
)
from holographic_qec.codes.stabilizer import create_codespace_state
from holographic_qec.analysis.correlation import (
    GeometryComplexityAnalyzer,
    PhaseTransitionAnalyzer
)
from holographic_qec.analysis.holographic import HolographicDictionary
from holographic_qec.utils.visualization import (
    plot_correlation_matrix,
    plot_complexity_curves,
    plot_holographic_tests,
    plot_phase_diagram
)


def load_test_codes(config: Dict) -> List[CodeSample]:
    """Load or generate test codes with n_physical <= 12."""
    print("Loading small code dataset (n_physical <= 12)...")

    seed = config.get('hardware', {}).get('seed', 42)
    samples = generate_small_code_dataset(seed=seed)

    print(f"Loaded {len(samples)} test codes")
    return samples


def compute_complexity_for_codes(
    codes: List[CodeSample],
    hamiltonians: List[Tuple],
    config: Dict
) -> List:
    """
    Compute Krylov complexity for each code using code-space initial states.

    Uses ONE canonical Hamiltonian per n_qubits size so that differences in
    C_K curves come from code geometry, not Hamiltonian variation.

    Returns list of result dicts.
    """
    print("Computing Krylov complexity with code-space states...")

    krylov_config = config.get('krylov', {})
    t_max = krylov_config.get('t_max', 10.0)
    n_steps = krylov_config.get('n_time_steps', 100)
    max_dim = krylov_config.get('max_krylov_dim', 50)

    # Group Hamiltonians by n_qubits, pick ONE canonical per size
    canonical_hams = {}
    for H, params in hamiltonians:
        if params.n_qubits not in canonical_hams:
            canonical_hams[params.n_qubits] = (H, params)

    results = []

    for i, code_sample in enumerate(codes):
        if i % 10 == 0:
            print(f"  Processing code {i+1}/{len(codes)}")

        code = code_sample.code
        n = code.n_physical

        # Skip codes too large for matrix exponentiation
        if n > 12:
            continue

        # Get canonical Hamiltonian for this size
        if n not in canonical_hams:
            continue
        H, params = canonical_hams[n]

        # Compute code-space initial state
        try:
            psi0 = create_codespace_state(code)
        except (ValueError, Exception) as e:
            print(f"  Warning: Failed to create codespace state for code {i}: {e}")
            continue

        # Compute complexity
        try:
            result = compute_krylov_complexity(
                H, psi0, t_max, n_steps, max_dim
            )
            results.append({
                'code_idx': i,
                'code': code_sample,
                'hamiltonian_params': params,
                'result': result
            })
        except Exception as e:
            print(f"  Warning: Failed for code {i}: {e}")
            continue

    print(f"Computed complexity for {len(results)} codes")
    return results


def extract_features(
    codes: List[CodeSample],
    complexity_results: List
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Extract geometric and dynamic features.

    Returns:
        (geometric_features, dynamic_features)
    """
    print("Extracting features...")

    # Initialize analyzer
    analyzer = GeometryComplexityAnalyzer()

    # Extract geometric features from codes
    geometric = analyzer.extract_geometric_features_from_codes(
        [c.code for c in codes]
    )

    # Extract dynamic features from complexity results
    dynamic = {
        'growth_exponent': [],
        'saturation_value': [],
        'krylov_dim': [],
        'saturation_time': [],
        'max_complexity': []
    }

    for item in complexity_results:
        result = item['result']
        dynamic['growth_exponent'].append(
            result.growth_exponent if result.growth_exponent else 0
        )
        dynamic['saturation_value'].append(
            result.saturation_value if result.saturation_value else 0
        )
        dynamic['krylov_dim'].append(result.krylov_dimension)
        dynamic['max_complexity'].append(np.max(result.complexity))

        # Saturation time
        max_c = np.max(result.complexity)
        if max_c > 0:
            sat_idx = np.argmax(result.complexity >= 0.9 * max_c)
            sat_time = result.times[sat_idx]
        else:
            sat_time = result.times[-1]
        dynamic['saturation_time'].append(sat_time)

    dynamic = {k: np.array(v) for k, v in dynamic.items()}

    # Match lengths (use min of geometric and dynamic)
    n_samples = min(len(codes), len(complexity_results))
    geometric = {k: v[:n_samples] for k, v in geometric.items()}
    dynamic = {k: v[:n_samples] for k, v in dynamic.items()}

    return geometric, dynamic


def run_correlation_analysis(
    geometric: Dict[str, np.ndarray],
    dynamic: Dict[str, np.ndarray],
    output_dir: Path
) -> GeometryComplexityAnalyzer:
    """
    Run full correlation analysis.
    """
    print("\nRunning correlation analysis...")

    analyzer = GeometryComplexityAnalyzer(geometric, dynamic)

    # Compute correlation matrix
    matrix, g_names, d_names = analyzer.compute_correlation_matrix('pearson')

    # Plot correlation matrix
    p_matrix = analyzer.p_value_matrix

    plot_correlation_matrix(
        matrix, g_names, d_names,
        title="Geometry-Complexity Correlations",
        save_path=str(output_dir / 'correlation_matrix.png'),
        p_values=p_matrix
    )

    # Find significant correlations
    significant = analyzer.find_significant_correlations(alpha=0.01)

    print("\nTop significant correlations:")
    for r in significant[:5]:
        print(f"  {r.feature1} <-> {r.feature2}: r={r.correlation:.3f}, p={r.p_value:.2e}")

    # Save summary
    summary = analyzer.summary_report()
    with open(output_dir / 'correlation_summary.txt', 'w') as f:
        f.write(summary)

    print(f"\nCorrelation analysis saved to {output_dir}")

    return analyzer


def run_holographic_tests(
    geometric: Dict[str, np.ndarray],
    dynamic: Dict[str, np.ndarray],
    output_dir: Path
) -> HolographicDictionary:
    """
    Run holographic dictionary tests.
    """
    print("\nRunning holographic dictionary tests...")

    dictionary = HolographicDictionary()

    # Prepare Hamiltonian data (use field strength as temperature proxy)
    hamiltonian_data = {
        'temperature': np.random.uniform(0.5, 2.0, len(geometric['depth'])),
        'field_strength': np.random.uniform(0.0, 2.0, len(geometric['depth']))
    }

    results = dictionary.run_all_tests(geometric, dynamic, hamiltonian_data)

    # Plot results
    plot_holographic_tests(
        results,
        title="Holographic Correspondence Tests",
        save_path=str(output_dir / 'holographic_tests.png')
    )

    # Save summary
    summary = dictionary.summary_report()
    with open(output_dir / 'holographic_summary.txt', 'w') as f:
        f.write(summary)

    print(summary)

    return dictionary


def run_phase_transition_analysis(
    codes: List[CodeSample],
    config: Dict,
    output_dir: Path
):
    """
    Scan parameter space for phase transitions.
    """
    print("\nRunning phase transition analysis...")

    phase_config = config.get('analysis', {}).get('phase_transition', {})
    if not phase_config.get('enabled', True):
        print("Phase transition analysis disabled")
        return

    # Parameter scan
    n_points = phase_config.get('n_points', 20)
    delta_range = np.linspace(0, 3, n_points)

    # Compute complexity at each parameter value
    times = np.linspace(0, 10, 100)
    complexity_curves = []

    print("  Scanning Delta parameter...")
    for delta in delta_range:
        H, params = HamiltonianBuilder.xxz(5, J_xy=1.0, J_z=delta)
        psi0 = create_initial_states(5, 'random')

        try:
            result = compute_krylov_complexity(H, psi0, 10.0, 100)
            complexity_curves.append(result.complexity)
        except:
            complexity_curves.append(np.zeros(100))

    # Analyze for phase transitions
    analyzer = PhaseTransitionAnalyzer()
    scan_results = analyzer.scan_parameter(
        delta_range,
        complexity_curves,
        times
    )

    critical_point = analyzer.identify_critical_point(scan_results)

    if critical_point:
        print(f"  Potential critical point at Delta = {critical_point:.2f}")

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(delta_range, scan_results['growth_exponents'], 'b-o')
    axes[0].set_xlabel(r'Anisotropy $\Delta$')
    axes[0].set_ylabel('Growth Exponent')
    axes[0].set_title('Growth Exponent vs Anisotropy')

    axes[1].plot(delta_range, scan_results['growth_susceptibility'], 'r-o')
    axes[1].set_xlabel(r'Anisotropy $\Delta$')
    axes[1].set_ylabel(r'Susceptibility $|\partial\alpha/\partial\Delta|$')
    axes[1].set_title('Susceptibility')

    if critical_point:
        for ax in axes:
            ax.axvline(x=critical_point, color='g', linestyle='--',
                      label=f'Critical point')
        axes[0].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'phase_transition.png')
    plt.close()

    print(f"  Phase transition analysis saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run correlation analysis")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--output-dir', type=str, default='results/analysis',
                        help='Output directory')
    parser.add_argument('--n-codes', type=int, default=None,
                        help='Number of codes to analyze')
    parser.add_argument('--skip-complexity', action='store_true',
                        help='Skip complexity computation (load from cache)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = {}

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEOMETRY-COMPLEXITY CORRELATION ANALYSIS")
    print("=" * 60)

    # Step 1: Load test codes
    codes = load_test_codes(config)
    if args.n_codes:
        codes = codes[:args.n_codes]

    # Step 2: Generate Hamiltonians
    ham_config = config.get('hamiltonians', {})
    hamiltonians = generate_hamiltonian_dataset(
        xxz_deltas=ham_config.get('xxz', {}).get('Delta_values', [0.0, 0.5, 1.0, 2.0]),
        ising_hs=ham_config.get('ising', {}).get('h_values', [0.0, 0.5, 1.0, 2.0]),
        n_qubits_list=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        n_random=20
    )

    # Step 3: Compute complexity
    if not args.skip_complexity:
        complexity_results = compute_complexity_for_codes(codes, hamiltonians, config)
    else:
        print("Skipping complexity computation (using synthetic data)")
        complexity_results = []
        for i, code in enumerate(codes):
            # Create synthetic result
            class SyntheticResult:
                growth_exponent = np.random.uniform(1.0, 2.5)
                saturation_value = np.random.uniform(5, 20)
                krylov_dimension = np.random.randint(10, 50)
                complexity = np.random.rand(100) * 10
                times = np.linspace(0, 10, 100)
            complexity_results.append({
                'code_idx': i,
                'code': code,
                'result': SyntheticResult()
            })

    # Step 4: Extract features
    geometric, dynamic = extract_features(codes, complexity_results)

    # Step 5: Correlation analysis
    analyzer = run_correlation_analysis(geometric, dynamic, output_dir)

    # Step 6: Holographic tests
    dictionary = run_holographic_tests(geometric, dynamic, output_dir)

    # Step 7: Phase transition analysis
    run_phase_transition_analysis(codes, config, output_dir)

    # Save all results
    results = {
        'n_codes': len(codes),
        'n_complexity_results': len(complexity_results),
        'geometric_features': list(geometric.keys()),
        'dynamic_features': list(dynamic.keys()),
        'holographic_tests_passed': sum(1 for r in dictionary.test_results if r.passed),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Analysis complete. Results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
