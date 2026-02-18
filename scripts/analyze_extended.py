#!/usr/bin/env python3
"""
Extended Correlation Analysis

Additional experiments to strengthen the geometry-complexity claim:
1. Partial correlations controlling for n_physical
2. Within-size correlations (fixed n_physical)
3. Robustness across multiple canonical Hamiltonians
4. DeepONet ablation summary table
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from holographic_qec.codes.dataset import generate_small_code_dataset, CodeSample
from holographic_qec.codes.stabilizer import create_codespace_state, pauli_weight
from holographic_qec.dynamics.krylov import compute_krylov_complexity
from holographic_qec.dynamics.hamiltonian import (
    HamiltonianBuilder,
    generate_hamiltonian_dataset,
)


def compute_complexity_dataset(
    codes: List[CodeSample],
    hamiltonians: List[Tuple],
    t_max: float = 10.0,
    n_steps: int = 100,
    max_dim: int = 50,
    label: str = ""
) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute C_K for all codes with one canonical Hamiltonian per size.

    Returns:
        valid_indices: indices of codes that succeeded
        geometric: dict of geometric feature arrays
        dynamic: dict of dynamic feature arrays
    """
    # Pick ONE canonical Hamiltonian per size
    canonical_hams = {}
    for H, params in hamiltonians:
        if params.n_qubits not in canonical_hams:
            canonical_hams[params.n_qubits] = (H, params)

    valid_indices = []
    geo = defaultdict(list)
    dyn = defaultdict(list)

    for i, sample in enumerate(codes):
        code = sample.code
        n = code.n_physical
        if n > 12 or n not in canonical_hams:
            continue

        H, params = canonical_hams[n]

        try:
            psi0 = create_codespace_state(code)
        except Exception:
            continue

        try:
            result = compute_krylov_complexity(H, psi0, t_max, n_steps, max_dim)
        except Exception:
            continue

        valid_indices.append(i)

        # Geometric features
        geo['n_physical'].append(n)
        geo['n_logical'].append(code.n_logical)
        geo['distance'].append(code.distance if code.distance else 0)
        geo['rate'].append(code.n_logical / code.n_physical)
        geo['avg_stabilizer_weight'].append(
            np.mean([pauli_weight(s) for s in code.stabilizers])
        )

        # Dynamic features
        dyn['growth_exponent'].append(
            result.growth_exponent if result.growth_exponent else 0
        )
        dyn['saturation_value'].append(
            result.saturation_value if result.saturation_value else 0
        )
        dyn['krylov_dim'].append(result.krylov_dimension)
        dyn['max_complexity'].append(np.max(result.complexity))

        max_c = np.max(result.complexity)
        if max_c > 0:
            sat_idx = np.argmax(result.complexity >= 0.9 * max_c)
            sat_time = result.times[sat_idx]
        else:
            sat_time = result.times[-1]
        dyn['saturation_time'].append(sat_time)

    geo = {k: np.array(v) for k, v in geo.items()}
    dyn = {k: np.array(v) for k, v in dyn.items()}

    if label:
        print(f"  [{label}] Computed C_K for {len(valid_indices)} codes")

    return valid_indices, geo, dyn


def partial_correlation(x, y, z):
    """
    Compute partial correlation between x and y, controlling for z.
    Uses linear regression to remove effect of z from both x and y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    # Residualize x and y on z
    slope_xz, intercept_xz, _, _, _ = stats.linregress(z, x)
    slope_yz, intercept_yz, _, _, _ = stats.linregress(z, y)

    x_resid = x - (slope_xz * z + intercept_xz)
    y_resid = y - (slope_yz * z + intercept_yz)

    r, p = stats.pearsonr(x_resid, y_resid)
    return r, p


# ============================================================
# Experiment 1: Partial correlations controlling for n_physical
# ============================================================
def run_partial_correlations(geo, dyn):
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Partial Correlations (controlling for n_physical)")
    print("=" * 60)

    z = geo['n_physical']
    geo_features = ['distance', 'rate', 'avg_stabilizer_weight']
    dyn_features = ['growth_exponent', 'saturation_value', 'krylov_dim',
                    'saturation_time', 'max_complexity']

    results = {}
    print(f"\n{'Geometric':<25} {'Dynamic':<20} {'raw r':>8} {'raw p':>10} {'partial r':>10} {'partial p':>12}")
    print("-" * 90)

    for gf in geo_features:
        for df in dyn_features:
            raw_r, raw_p = stats.pearsonr(geo[gf], dyn[df])
            part_r, part_p = partial_correlation(geo[gf], dyn[df], z)

            sig_raw = "***" if raw_p < 0.001 else "**" if raw_p < 0.01 else "*" if raw_p < 0.05 else ""
            sig_part = "***" if part_p < 0.001 else "**" if part_p < 0.01 else "*" if part_p < 0.05 else ""

            print(f"{gf:<25} {df:<20} {raw_r:>7.3f}{sig_raw:<3} {raw_p:>10.2e} {part_r:>9.3f}{sig_part:<3} {part_p:>12.2e}")

            results[f"{gf}__{df}"] = {
                'raw_r': float(raw_r), 'raw_p': float(raw_p),
                'partial_r': float(part_r), 'partial_p': float(part_p)
            }

    return results


# ============================================================
# Experiment 2: Within-size correlations
# ============================================================
def run_within_size_analysis(codes, geo, dyn):
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Within-Size Correlations")
    print("=" * 60)

    results = {}
    sizes = sorted(set(geo['n_physical'].astype(int)))

    geo_features = ['distance', 'rate', 'avg_stabilizer_weight']
    dyn_features = ['growth_exponent', 'saturation_value', 'krylov_dim',
                    'saturation_time', 'max_complexity']

    for n in sizes:
        mask = geo['n_physical'] == n
        n_codes = int(np.sum(mask))
        if n_codes < 10:
            print(f"\nn_physical={n}: only {n_codes} codes, skipping")
            continue

        # Check variance in distance
        dist_vals = geo['distance'][mask]
        unique_dists = len(set(dist_vals))

        print(f"\nn_physical={n}: {n_codes} codes, {unique_dists} unique distances")
        results[str(n)] = {'n_codes': n_codes, 'correlations': {}}

        for gf in geo_features:
            gvals = geo[gf][mask]
            if np.std(gvals) < 1e-10:
                continue
            for df in dyn_features:
                dvals = dyn[df][mask]
                if np.std(dvals) < 1e-10:
                    continue
                r, p = stats.pearsonr(gvals, dvals)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                if abs(r) > 0.15 or p < 0.05:
                    print(f"  {gf:<25} <-> {df:<20}: r={r:>7.3f}{sig}, p={p:.2e}")
                results[str(n)]['correlations'][f"{gf}__{df}"] = {
                    'r': float(r), 'p': float(p)
                }

    return results


# ============================================================
# Experiment 3: Multiple Hamiltonians robustness check
# ============================================================
def run_multi_hamiltonian_robustness(codes):
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Robustness Across Multiple Hamiltonians")
    print("=" * 60)

    # Generate several different Hamiltonian sets
    ham_configs = [
        ("XXZ Delta=0.0", {'xxz_deltas': [0.0], 'ising_hs': [], 'n_qubits_list': [4,5,6,7,8,9,10,11,12], 'n_random': 0}),
        ("XXZ Delta=1.0", {'xxz_deltas': [1.0], 'ising_hs': [], 'n_qubits_list': [4,5,6,7,8,9,10,11,12], 'n_random': 0}),
        ("XXZ Delta=2.0", {'xxz_deltas': [2.0], 'ising_hs': [], 'n_qubits_list': [4,5,6,7,8,9,10,11,12], 'n_random': 0}),
        ("Ising h=1.0", {'xxz_deltas': [], 'ising_hs': [1.0], 'n_qubits_list': [4,5,6,7,8,9,10,11,12], 'n_random': 0}),
        ("Random", {'xxz_deltas': [], 'ising_hs': [], 'n_qubits_list': [4,5,6,7,8,9,10,11,12], 'n_random': 10}),
    ]

    # Key correlations to track
    key_pairs = [
        ('distance', 'growth_exponent'),
        ('distance', 'max_complexity'),
        ('rate', 'saturation_value'),
    ]

    results = {}
    print(f"\n{'Hamiltonian':<20}", end="")
    for gf, df in key_pairs:
        print(f"  {gf[:8]}-{df[:8]:>10}", end="")
    print(f"  {'N':>5}")
    print("-" * 80)

    for label, kwargs in ham_configs:
        hams = generate_hamiltonian_dataset(**kwargs)
        if not hams:
            continue

        _, geo, dyn = compute_complexity_dataset(codes, hams, label=label)
        if len(geo.get('distance', [])) < 20:
            print(f"{label:<20}  insufficient data ({len(geo.get('distance', []))} codes)")
            continue

        results[label] = {}
        print(f"{label:<20}", end="")
        for gf, df in key_pairs:
            r, p = stats.pearsonr(geo[gf], dyn[df])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {r:>7.3f}{sig:<3}      ", end="")
            results[label][f"{gf}__{df}"] = {'r': float(r), 'p': float(p)}
        print(f"  {len(geo['distance']):>5}")

    return results


# ============================================================
# Experiment 4: DeepONet ablation table
# ============================================================
def format_ablation_table():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: DeepONet Ablation Summary")
    print("=" * 60)

    # Load results
    results_dir = Path("results")
    new_results_path = results_dir / "deeponet_results.json"

    # Find the previous (zeroed embeddings) result
    old_results_path = results_dir / "experiment" / "20260208_103112" / "deeponet_results.json"

    table = {}

    if old_results_path.exists():
        with open(old_results_path) as f:
            old = json.load(f)
        table['Zeroed embeddings (old)'] = old['final_metrics']
        table['Zeroed embeddings (old)']['best_epoch'] = old['best_epoch']
        table['Zeroed embeddings (old)']['final_train_loss'] = old['history']['train_loss'][-1]

    if new_results_path.exists():
        with open(new_results_path) as f:
            new = json.load(f)
        table['Code-space states (new)'] = new['final_metrics']
        table['Code-space states (new)']['best_epoch'] = new['best_epoch']
        table['Code-space states (new)']['final_train_loss'] = new['history']['train_loss'][-1]

    print(f"\n{'Configuration':<30} {'R²':>8} {'MSE':>10} {'RMSE':>8} {'Train Loss':>12} {'Best Epoch':>12}")
    print("-" * 85)

    for config_name, metrics in table.items():
        print(f"{config_name:<30} {metrics['r2']:>8.3f} {metrics['mse']:>10.2f} {metrics['rmse']:>8.3f} "
              f"{metrics['final_train_loss']:>12.4f} {metrics['best_epoch']:>12d}")

    return table


# ============================================================
# Main
# ============================================================
def main():
    output_dir = Path("results/analysis/extended")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXTENDED CORRELATION ANALYSIS")
    print("=" * 60)

    # Generate codes and base Hamiltonians
    print("\nGenerating codes and Hamiltonians...")
    codes = generate_small_code_dataset(seed=42)
    hamiltonians = generate_hamiltonian_dataset(
        xxz_deltas=[0.0, 0.5, 1.0, 2.0],
        ising_hs=[0.0, 0.5, 1.0, 2.0],
        n_qubits_list=[4, 5, 6, 7, 8, 9, 10, 11, 12],
        n_random=20
    )

    # Compute base dataset
    print("\nComputing base complexity dataset...")
    valid_idx, geo, dyn = compute_complexity_dataset(
        codes, hamiltonians, label="base"
    )

    all_results = {}

    # Experiment 1
    all_results['partial_correlations'] = run_partial_correlations(geo, dyn)

    # Experiment 2
    all_results['within_size'] = run_within_size_analysis(codes, geo, dyn)

    # Experiment 3
    all_results['multi_hamiltonian'] = run_multi_hamiltonian_robustness(codes)

    # Experiment 4
    all_results['ablation'] = format_ablation_table()

    # Save all results
    with open(output_dir / 'extended_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print(f"All results saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
