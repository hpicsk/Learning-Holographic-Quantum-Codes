#!/usr/bin/env python3
"""
Full Experiment Runner

Complete experimental pipeline for paper:
1. Train GNN on code dataset
2. Train DeepONet on complexity dynamics
3. Run correlation analysis
4. Run holographic dictionary tests
5. Generate publication figures

Usage:
    python run_experiments.py --config configs/default.yaml
    python run_experiments.py --preset debug  # Quick test run
    python run_experiments.py --preset full   # Full research run
    python run_experiments.py --resume results/experiment/20240101_120000  # Resume from checkpoint
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_gnn_training(config: Dict, output_dir: Path) -> Dict[str, Any]:
    """Run GNN training pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 1: GNN TRAINING")
    print("=" * 60)

    from scripts.train_gnn import train_gnn

    # Update output directory
    config['output'] = config.get('output', {})
    config['output']['base_dir'] = str(output_dir)

    model, results = train_gnn(config)

    # Save checkpoint path
    results['checkpoint_path'] = str(output_dir / 'checkpoints' / 'best_model.pt')

    return results


def run_deeponet_training(
    config: Dict,
    output_dir: Path,
    gnn_checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Run DeepONet training pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 2: DeepONet TRAINING")
    print("=" * 60)

    from scripts.train_deeponet import train_deeponet

    # Update config
    config['output'] = config.get('output', {})
    config['output']['base_dir'] = str(output_dir)

    if gnn_checkpoint:
        config['gnn_checkpoint'] = gnn_checkpoint

    model, results = train_deeponet(config)

    results['checkpoint_path'] = str(output_dir / 'checkpoints' / 'best_deeponet.pt')

    return results


def run_analysis(config: Dict, output_dir: Path) -> Dict[str, Any]:
    """Run full analysis pipeline."""
    print("\n" + "=" * 60)
    print("PHASE 3: ANALYSIS")
    print("=" * 60)

    analysis_dir = output_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Import analysis components
    from holographic_qec.codes.dataset import HolographicCodeDataset
    from holographic_qec.dynamics.krylov import compute_krylov_complexity
    from holographic_qec.dynamics.hamiltonian import (
        generate_hamiltonian_dataset,
        create_initial_states
    )
    from holographic_qec.analysis.correlation import GeometryComplexityAnalyzer
    from holographic_qec.analysis.holographic import HolographicDictionary
    from holographic_qec.utils.visualization import (
        plot_correlation_matrix,
        plot_holographic_tests,
        plot_complexity_curves
    )

    # Generate test data
    print("Generating test data...")
    dataset_config = config.get('dataset', {})

    # Use smaller dataset for analysis
    dataset = HolographicCodeDataset(
        n_happy=min(dataset_config.get('n_happy', 100), 50),
        n_ldpc=min(dataset_config.get('n_ldpc', 50), 20),
        n_random=min(dataset_config.get('n_random', 20), 10)
    )
    dataset.generate(parallel=False)
    _, _, test_codes = dataset.get_splits()

    # Generate Hamiltonians with expanded parameter ranges
    hamiltonians = generate_hamiltonian_dataset(
        xxz_deltas=[0.0, 0.5, 1.0, 2.0],
        ising_hs=[0.0, 0.5, 1.0, 1.5, 2.0],
        n_qubits_list=[4, 5, 6],
        n_random=20,
        xxz_fields=[0.0, 0.5, 1.0]  # Add field variation to XXZ
    )

    # Compute complexity for subset
    print("Computing Krylov complexity...")
    results_list = []
    n_samples = min(200, len(test_codes))

    for i in range(n_samples):
        code = test_codes[i]
        H, params = hamiltonians[i % len(hamiltonians)]

        psi0 = create_initial_states(params.n_qubits, 'random')

        try:
            result = compute_krylov_complexity(H, psi0, 10.0, 100)
            results_list.append({
                'code': code,
                'params': params,
                'result': result
            })
        except Exception as e:
            print(f"  Warning: Failed for sample {i}: {e}")

    # Extract features
    print("Extracting features...")
    analyzer = GeometryComplexityAnalyzer()
    geometric = analyzer.extract_geometric_features_from_codes(
        [r['code'].code for r in results_list]
    )

    dynamic = {
        'growth_exponent': np.array([
            r['result'].growth_exponent or 0 for r in results_list
        ]),
        'saturation_value': np.array([
            r['result'].saturation_value or 0 for r in results_list
        ]),
        'krylov_dim': np.array([
            r['result'].krylov_dimension for r in results_list
        ]).astype(float)
    }

    # Extract Hamiltonian parameters for holographic tests
    hamiltonian_data = {
        'field_strength': np.array([
            r['params'].field_strength for r in results_list
        ]),
        'temperature': np.array([
            # Use field_strength as effective temperature proxy (T ~ g for Ising)
            # Add anisotropy contribution for meaningful variation
            max(r['params'].field_strength, 0.1) + 0.1 * r['params'].anisotropy
            for r in results_list
        ])
    }

    analyzer.set_dynamic_features(dynamic)

    # Compute correlations
    print("Computing correlations...")
    matrix, g_names, d_names = analyzer.compute_correlation_matrix('pearson')

    plot_correlation_matrix(
        matrix, g_names, d_names,
        title="Geometry-Complexity Correlations",
        save_path=str(analysis_dir / 'correlation_matrix.png'),
        p_values=analyzer.p_value_matrix
    )

    # Run holographic tests
    print("Running holographic tests...")
    dictionary = HolographicDictionary()
    holo_results = dictionary.run_all_tests(geometric, dynamic, hamiltonian_data)

    plot_holographic_tests(
        holo_results,
        save_path=str(analysis_dir / 'holographic_tests.png')
    )

    # Plot example complexity curves
    print("Generating complexity curve plots...")
    times = results_list[0]['result'].times
    curves = [r['result'].complexity for r in results_list[:5]]
    labels = [f"Code {i+1}" for i in range(len(curves))]

    plot_complexity_curves(
        times, curves, labels,
        title="Krylov Complexity Dynamics",
        save_path=str(analysis_dir / 'complexity_curves.png')
    )

    # Compile results
    significant = analyzer.find_significant_correlations(alpha=0.05)

    analysis_results = {
        'n_samples': len(results_list),
        'n_significant_correlations': len(significant),
        'holographic_tests_passed': sum(1 for r in holo_results if r.passed),
        'holographic_tests_total': len(holo_results),
        'top_correlations': [
            {
                'features': (r.feature1, r.feature2),
                'correlation': r.correlation,
                'p_value': r.p_value
            }
            for r in significant[:5]
        ]
    }

    # Save summaries
    with open(analysis_dir / 'correlation_summary.txt', 'w') as f:
        f.write(analyzer.summary_report())

    with open(analysis_dir / 'holographic_summary.txt', 'w') as f:
        f.write(dictionary.summary_report())

    print(f"Analysis results saved to {analysis_dir}")

    return analysis_results


def generate_paper_figures(output_dir: Path, results: Dict[str, Any]):
    """Generate publication-quality figures."""
    print("\n" + "=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # The individual analysis steps already generate most figures
    # This function could combine them or add additional plots

    print(f"Figures saved to {figures_dir}")


def detect_completed_phases(output_dir: Path) -> Dict[str, bool]:
    """
    Detect which phases have completed successfully in a previous run.

    Returns a dict with keys: 'gnn', 'deeponet', 'analysis', 'figures'
    """
    completed = {
        'gnn': False,
        'deeponet': False,
        'analysis': False,
        'figures': False
    }

    # Check for GNN completion
    gnn_checkpoint = output_dir / 'checkpoints' / 'best_model.pt'
    gnn_results = output_dir / 'gnn_results.json'
    if gnn_checkpoint.exists() or gnn_results.exists():
        completed['gnn'] = True

    # Check for DeepONet completion
    deeponet_checkpoint = output_dir / 'checkpoints' / 'best_deeponet.pt'
    deeponet_results = output_dir / 'deeponet_results.json'
    if deeponet_checkpoint.exists() or deeponet_results.exists():
        completed['deeponet'] = True

    # Check for Analysis completion
    analysis_dir = output_dir / 'analysis'
    if analysis_dir.exists():
        analysis_files = list(analysis_dir.glob('*.txt')) + list(analysis_dir.glob('*.png'))
        if len(analysis_files) >= 3:  # correlation_summary, holographic_summary, at least one plot
            completed['analysis'] = True

    # Check for figures
    figures_dir = output_dir / 'figures'
    if figures_dir.exists() and any(figures_dir.iterdir()):
        completed['figures'] = True

    return completed


def load_previous_results(output_dir: Path) -> Dict[str, Any]:
    """Load results from a previous experiment run."""
    results = {}

    # Load main experiment results if available
    experiment_results_path = output_dir / 'experiment_results.json'
    if experiment_results_path.exists():
        with open(experiment_results_path) as f:
            results = json.load(f)

    # Load individual phase results if not in main results
    if 'gnn' not in results or 'error' in results.get('gnn', {}):
        gnn_results_path = output_dir / 'gnn_results.json'
        if gnn_results_path.exists():
            with open(gnn_results_path) as f:
                results['gnn'] = json.load(f)
            results['gnn']['checkpoint_path'] = str(output_dir / 'checkpoints' / 'best_model.pt')

    if 'deeponet' not in results or 'error' in results.get('deeponet', {}):
        deeponet_results_path = output_dir / 'deeponet_results.json'
        if deeponet_results_path.exists():
            with open(deeponet_results_path) as f:
                results['deeponet'] = json.load(f)
            results['deeponet']['checkpoint_path'] = str(output_dir / 'checkpoints' / 'best_deeponet.pt')

    return results


def save_phase_state(output_dir: Path, phase: str, status: str):
    """Save the state of a phase for resume tracking."""
    state_file = output_dir / 'experiment_state.json'

    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {'phases': {}}

    state['phases'][phase] = {
        'status': status,
        'timestamp': datetime.now().isoformat()
    }
    state['last_updated'] = datetime.now().isoformat()

    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['debug', 'full', 'analysis_only'],
                        help='Use preset configuration')
    parser.add_argument('--output-dir', type=str, default='results/experiment',
                        help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous experiment directory')
    parser.add_argument('--skip-gnn', action='store_true',
                        help='Skip GNN training')
    parser.add_argument('--skip-deeponet', action='store_true',
                        help='Skip DeepONet training')
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip analysis')
    parser.add_argument('--force-rerun', action='store_true',
                        help='Force rerun of all phases even if completed (use with --resume)')

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found: {config_path}, using defaults")
        config = {}

    # Apply preset
    if args.preset:
        if args.preset == 'debug':
            config.setdefault('dataset', {})['n_happy'] = 20
            config.setdefault('dataset', {})['n_ldpc'] = 10
            config.setdefault('dataset', {})['n_random'] = 5
            config.setdefault('gnn', {}).setdefault('training', {})['epochs'] = 10
            config.setdefault('deeponet', {}).setdefault('training', {})['epochs'] = 10
        elif args.preset == 'analysis_only':
            args.skip_gnn = True
            args.skip_deeponet = True

    # Handle resume or new experiment
    resuming = False
    completed_phases = {'gnn': False, 'deeponet': False, 'analysis': False, 'figures': False}

    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"Error: Resume directory does not exist: {resume_dir}")
            sys.exit(1)

        output_dir = resume_dir
        resuming = True

        # Detect what's already completed
        completed_phases = detect_completed_phases(output_dir)

        # Load config from previous run if exists
        prev_config_path = output_dir / 'config.yaml'
        if prev_config_path.exists():
            with open(prev_config_path) as f:
                prev_config = yaml.safe_load(f)
            # Merge: previous config as base, current args override
            for key, value in prev_config.items():
                if key not in config:
                    config[key] = value

        print("=" * 60)
        print("RESUMING EXPERIMENT")
        print("=" * 60)
        print(f"Resume directory: {output_dir}")
        print(f"Completed phases:")
        for phase, done in completed_phases.items():
            status = "DONE" if done else "PENDING"
            print(f"  - {phase}: {status}")
        print("=" * 60)

        if not args.force_rerun:
            # Skip completed phases unless force-rerun
            if completed_phases['gnn']:
                args.skip_gnn = True
            if completed_phases['deeponet']:
                args.skip_deeponet = True
            if completed_phases['analysis']:
                args.skip_analysis = True
    else:
        # Setup new output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(args.output_dir) / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    if not resuming:
        print("=" * 60)
        print("HOLOGRAPHIC QEC EXPERIMENT PIPELINE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Config: {args.config}")
        print(f"Preset: {args.preset or 'none'}")
        print("=" * 60)

    start_time = time.time()

    # Load previous results if resuming
    if resuming:
        all_results = load_previous_results(output_dir)
        print(f"\nLoaded {len(all_results)} previous result sections")
    else:
        all_results = {}

    # Phase 1: GNN Training
    if not args.skip_gnn:
        save_phase_state(output_dir, 'gnn', 'running')
        try:
            gnn_results = run_gnn_training(config, output_dir)
            all_results['gnn'] = gnn_results
            save_phase_state(output_dir, 'gnn', 'completed')
        except Exception as e:
            print(f"GNN training failed: {e}")
            all_results['gnn'] = {'error': str(e)}
            save_phase_state(output_dir, 'gnn', 'failed')
    else:
        if resuming and completed_phases.get('gnn'):
            print("\nSkipping GNN training (already completed)")
        else:
            print("\nSkipping GNN training")

    # Phase 2: DeepONet Training
    if not args.skip_deeponet:
        save_phase_state(output_dir, 'deeponet', 'running')
        try:
            gnn_checkpoint = all_results.get('gnn', {}).get('checkpoint_path')
            # If resuming and GNN was completed, find the checkpoint
            if not gnn_checkpoint and resuming:
                checkpoint_path = output_dir / 'checkpoints' / 'best_model.pt'
                if checkpoint_path.exists():
                    gnn_checkpoint = str(checkpoint_path)
            deeponet_results = run_deeponet_training(config, output_dir, gnn_checkpoint)
            all_results['deeponet'] = deeponet_results
            save_phase_state(output_dir, 'deeponet', 'completed')
        except Exception as e:
            print(f"DeepONet training failed: {e}")
            all_results['deeponet'] = {'error': str(e)}
            save_phase_state(output_dir, 'deeponet', 'failed')
    else:
        if resuming and completed_phases.get('deeponet'):
            print("\nSkipping DeepONet training (already completed)")
        else:
            print("\nSkipping DeepONet training")

    # Phase 3: Analysis
    if not args.skip_analysis:
        save_phase_state(output_dir, 'analysis', 'running')
        try:
            analysis_results = run_analysis(config, output_dir)
            all_results['analysis'] = analysis_results
            save_phase_state(output_dir, 'analysis', 'completed')
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            all_results['analysis'] = {'error': str(e)}
            save_phase_state(output_dir, 'analysis', 'failed')
    else:
        if resuming and completed_phases.get('analysis'):
            print("\nSkipping analysis (already completed)")
        else:
            print("\nSkipping analysis")

    # Generate final figures
    save_phase_state(output_dir, 'figures', 'running')
    try:
        generate_paper_figures(output_dir, all_results)
        save_phase_state(output_dir, 'figures', 'completed')
    except Exception as e:
        print(f"Figure generation failed: {e}")
        save_phase_state(output_dir, 'figures', 'failed')

    # Save all results
    elapsed_time = time.time() - start_time

    # Update metadata
    if 'metadata' not in all_results:
        all_results['metadata'] = {}

    all_results['metadata'].update({
        'last_run_timestamp': datetime.now().isoformat(),
        'last_run_elapsed_seconds': elapsed_time,
        'config_path': str(args.config),
        'preset': args.preset,
        'resumed': resuming
    })

    # Keep original timestamp if resuming
    if not resuming:
        all_results['metadata']['timestamp'] = datetime.now().isoformat()
        all_results['metadata']['elapsed_seconds'] = elapsed_time

    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    if resuming:
        print("EXPERIMENT RESUMED AND COMPLETE")
    else:
        print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"This run time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"Results saved to: {output_dir}")

    if 'gnn' in all_results and 'error' not in all_results['gnn']:
        print(f"\nGNN Results:")
        if 'test_metrics' in all_results['gnn']:
            metrics = all_results['gnn']['test_metrics']
            print(f"  Distance MAE: {metrics.get('distance_mae', 'N/A'):.3f}")
            print(f"  Distance Relative Error: {metrics.get('distance_rel_error', 'N/A'):.2f}%")

    if 'deeponet' in all_results and 'error' not in all_results['deeponet']:
        print(f"\nDeepONet Results:")
        if 'final_metrics' in all_results['deeponet']:
            metrics = all_results['deeponet']['final_metrics']
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  R² Score: {metrics.get('r2', 'N/A'):.4f}")

    if 'analysis' in all_results and 'error' not in all_results['analysis']:
        print(f"\nAnalysis Results:")
        analysis = all_results['analysis']
        print(f"  Significant correlations: {analysis.get('n_significant_correlations', 'N/A')}")
        print(f"  Holographic tests passed: {analysis.get('holographic_tests_passed', 'N/A')}/{analysis.get('holographic_tests_total', 'N/A')}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
