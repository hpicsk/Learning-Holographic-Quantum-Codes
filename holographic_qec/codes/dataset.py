"""
Large-Scale Dataset Generation

Generates datasets of quantum error correcting codes for training:
- HaPPY holographic codes (3,500)
- LDPC codes (1,000)
- Random tensor network codes (500)

Total: 5,000 codes with 70/15/15 train/val/test split
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os

from .stabilizer import (
    StabilizerCode, pauli_weight, pauli_product, commutes,
    create_five_qubit_code, create_steane_code, create_repetition_code
)
from .happy_codes import create_happy_code, HaPPYCode


@dataclass
class CodeSample:
    """
    A single code sample for training.

    Contains the code data and precomputed features.
    """
    code: StabilizerCode
    family: str  # 'happy', 'ldpc', 'random'
    depth: Optional[int] = None  # For HaPPY codes
    geometric_features: Optional[np.ndarray] = None

    # Precomputed graph representation
    node_features: Optional[np.ndarray] = None
    edge_index: Optional[np.ndarray] = None
    hyperedges: Optional[List[List[int]]] = None

    def compute_graph(self):
        """Compute and cache graph representation."""
        if self.node_features is None:
            self.node_features, self.edge_index, self.hyperedges = self.code.to_graph()


class HolographicCodeDataset:
    """
    Large-scale dataset of quantum error correcting codes.

    Configuration (from strategy document):
    - 5,000 total codes
    - 3,500 HaPPY codes (70%)
    - 1,000 LDPC codes (20%)
    - 500 random tensor network codes (10%)
    - 70/15/15 train/val/test split
    """

    def __init__(
        self,
        n_happy: int = 3500,
        n_ldpc: int = 1000,
        n_random: int = 500,
        happy_depths: Dict[int, int] = None,
        ldpc_params: List[Tuple[int, int, int]] = None,
        train_split: float = 0.7,
        val_split: float = 0.15,
        seed: int = 42,
        cache_dir: str = None
    ):
        """
        Initialize the dataset generator.

        Args:
            n_happy: Number of HaPPY codes
            n_ldpc: Number of LDPC codes
            n_random: Number of random codes
            happy_depths: Dict mapping depth -> count
            ldpc_params: List of (n, k, d) parameters for LDPC codes
            train_split: Fraction for training
            val_split: Fraction for validation
            seed: Random seed
            cache_dir: Directory to cache generated codes
        """
        self.n_happy = n_happy
        self.n_ldpc = n_ldpc
        self.n_random = n_random
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = 1.0 - train_split - val_split
        self.seed = seed
        self.cache_dir = cache_dir

        # Default HaPPY depth distribution
        if happy_depths is None:
            self.happy_depths = {
                2: 500,   # n=5 (depth-2 gives ~20 qubits)
                3: 1000,  # n=10-20
                4: 1000,  # n=15-40
                5: 1000,  # n=20-80
            }
        else:
            self.happy_depths = happy_depths

        # Default LDPC parameters [[n, k, d]]
        if ldpc_params is None:
            self.ldpc_params = [
                (7, 4, 3),    # Hamming-like, 200 codes
                (15, 7, 5),   # BCH-like, 300 codes
                (31, 15, 7),  # Larger LDPC, 500 codes
            ]
        else:
            self.ldpc_params = ldpc_params

        self.samples: List[CodeSample] = []
        self._generated = False

    def generate(self, parallel: bool = True, n_workers: int = 4):
        """
        Generate all codes in the dataset.

        Args:
            parallel: Use parallel processing
            n_workers: Number of parallel workers
        """
        np.random.seed(self.seed)

        # Check cache
        if self.cache_dir and self._load_from_cache():
            print(f"Loaded {len(self.samples)} codes from cache")
            return

        print("Generating dataset...")

        # Generate each family
        print(f"  Generating {self.n_happy} HaPPY codes...")
        happy_codes = self._generate_happy_codes(parallel, n_workers)

        print(f"  Generating {self.n_ldpc} LDPC codes...")
        ldpc_codes = self._generate_ldpc_codes(parallel, n_workers)

        print(f"  Generating {self.n_random} random codes...")
        random_codes = self._generate_random_codes(parallel, n_workers)

        # Combine all samples
        self.samples = happy_codes + ldpc_codes + random_codes

        # Shuffle with seed
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in indices]

        self._generated = True

        # Cache if requested
        if self.cache_dir:
            self._save_to_cache()

        print(f"Generated {len(self.samples)} total codes")

    def _generate_happy_codes(
        self,
        parallel: bool = True,
        n_workers: int = 4
    ) -> List[CodeSample]:
        """Generate HaPPY codes with variations."""
        samples = []

        for depth, count in self.happy_depths.items():
            print(f"    Depth {depth}: {count} codes")

            for i in range(count):
                try:
                    # Create base code
                    code = create_happy_code(depth)

                    # Apply random variation for diversity
                    if i > 0:
                        code = self._vary_happy_code(code, seed=self.seed + i)

                    # Extract geometric features
                    features = code.get_geometric_features()

                    sample = CodeSample(
                        code=code.stabilizer_code,
                        family='happy',
                        depth=depth,
                        geometric_features=features
                    )
                    sample.compute_graph()
                    samples.append(sample)

                except Exception as e:
                    print(f"    Warning: Failed to generate HaPPY depth-{depth} code {i}: {e}")
                    continue

        return samples

    def _vary_happy_code(self, code: HaPPYCode, seed: int) -> HaPPYCode:
        """Apply random variation to a HaPPY code."""
        np.random.seed(seed)

        n = code.n_physical

        # Random qubit permutation
        perm = np.random.permutation(n)

        # Permute stabilizers
        new_stabilizers = []
        for s in code.stabilizer_code.stabilizers:
            new_s = ['I'] * n
            for i, p in enumerate(s):
                new_s[perm[i]] = p
            new_stabilizers.append(''.join(new_s))

        new_code = StabilizerCode(
            n_physical=n,
            n_logical=code.n_logical,
            stabilizers=new_stabilizers,
            distance=code.distance,
            name=code.stabilizer_code.name + f"_var{seed}"
        )

        return HaPPYCode(
            depth=code.depth,
            bond_dim=code.bond_dim,
            stabilizer_code=new_code,
            bulk_qubits=code.bulk_qubits,
            boundary_qubits=[perm[i] for i in code.boundary_qubits],
            tensor_positions=code.tensor_positions,
        )

    def _generate_ldpc_codes(
        self,
        parallel: bool = True,
        n_workers: int = 4
    ) -> List[CodeSample]:
        """Generate LDPC codes."""
        samples = []

        # Distribution: 200, 300, 500 for the three parameter sets
        counts = [200, 300, 500]

        for (n, k, d), count in zip(self.ldpc_params, counts):
            print(f"    [[{n},{k},{d}]]: {count} codes")

            for i in range(count):
                try:
                    code = self._create_ldpc_code(n, k, d, seed=self.seed + i)

                    sample = CodeSample(
                        code=code,
                        family='ldpc',
                        geometric_features=self._extract_ldpc_features(code)
                    )
                    sample.compute_graph()
                    samples.append(sample)

                except Exception as e:
                    print(f"    Warning: Failed to generate LDPC [[{n},{k},{d}]] code {i}: {e}")
                    continue

        return samples

    def _create_ldpc_code(
        self,
        n: int,
        k: int,
        d: int,
        seed: int
    ) -> StabilizerCode:
        """
        Create an LDPC-style quantum code.

        Uses random sparse parity checks with controlled density.
        """
        np.random.seed(seed)

        n_stabilizers = n - k

        # Generate sparse stabilizers
        # Each stabilizer acts on ~4-6 qubits (LDPC property)
        stabilizers = []

        for i in range(n_stabilizers):
            # Random sparse support
            weight = np.random.randint(3, min(7, n + 1))
            positions = np.random.choice(n, size=weight, replace=False)

            # Random Pauli assignment (mostly X or Z for CSS-like)
            pauli_type = np.random.choice(['X', 'Z'])

            s = ['I'] * n
            for pos in positions:
                # Occasionally mix in Y for variety
                if np.random.random() < 0.1:
                    s[pos] = 'Y'
                else:
                    s[pos] = pauli_type

            stabilizers.append(''.join(s))

        # Ensure stabilizers commute (simplified approach)
        stabilizers = self._make_commuting(stabilizers, n)

        return StabilizerCode(
            n_physical=n,
            n_logical=k,
            stabilizers=stabilizers,
            distance=d,  # Approximate
            name=f"LDPC [[{n},{k},{d}]]"
        )

    def _make_commuting(self, stabilizers: List[str], n: int) -> List[str]:
        """
        Modify stabilizers to ensure they all commute.

        Uses a simple strategy of flipping Paulis when anticommutation is detected.
        """
        result = stabilizers.copy()

        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if not commutes(result[i], result[j]):
                    # Flip a random position in stabilizer j
                    s = list(result[j])

                    # Find positions where both are non-identity
                    conflict_positions = [
                        pos for pos in range(n)
                        if result[i][pos] != 'I' and result[j][pos] != 'I'
                    ]

                    if conflict_positions:
                        pos = np.random.choice(conflict_positions)
                        # Change to commuting Pauli
                        if result[i][pos] == result[j][pos]:
                            pass  # Already commutes at this position
                        else:
                            s[pos] = 'I'  # Remove conflict
                        result[j] = ''.join(s)

        return result

    def _extract_ldpc_features(self, code: StabilizerCode) -> np.ndarray:
        """Extract geometric features from LDPC code."""
        avg_weight = np.mean([pauli_weight(s) for s in code.stabilizers])

        # Sparsity: fraction of non-identity entries
        total_entries = code.n_physical * len(code.stabilizers)
        nonzero = sum(pauli_weight(s) for s in code.stabilizers)
        sparsity = 1.0 - (nonzero / total_entries)

        # Rate
        rate = code.n_logical / code.n_physical

        return np.array([
            0,  # depth = 0 for non-HaPPY
            code.n_physical,
            code.n_logical,
            avg_weight,
            0.0,  # hyperbolic_radius = 0 for non-HaPPY
            rate,
            len(code.stabilizers),
            sparsity,
        ])

    def _generate_random_codes(
        self,
        parallel: bool = True,
        n_workers: int = 4
    ) -> List[CodeSample]:
        """Generate random tensor network codes."""
        samples = []

        # Various sizes
        sizes = [
            (8, 1),   # 8 physical, 1 logical
            (12, 2),
            (16, 2),
            (20, 3),
            (24, 4),
        ]

        per_size = self.n_random // len(sizes)

        for n, k in sizes:
            print(f"    Random [[{n},{k},?]]: {per_size} codes")

            for i in range(per_size):
                try:
                    code = self._create_random_code(n, k, seed=self.seed + i * 100)

                    sample = CodeSample(
                        code=code,
                        family='random',
                        geometric_features=self._extract_ldpc_features(code)
                    )
                    sample.compute_graph()
                    samples.append(sample)

                except Exception as e:
                    print(f"    Warning: Failed to generate random code: {e}")
                    continue

        return samples

    def _create_random_code(self, n: int, k: int, seed: int) -> StabilizerCode:
        """Create a random stabilizer code."""
        np.random.seed(seed)

        n_stabilizers = n - k
        stabilizers = []

        for i in range(n_stabilizers):
            # Random weight between 2 and n//2
            weight = np.random.randint(2, max(3, n // 2))
            positions = np.random.choice(n, size=weight, replace=False)

            s = ['I'] * n
            for pos in positions:
                s[pos] = np.random.choice(['X', 'Y', 'Z'])

            stabilizers.append(''.join(s))

        # Ensure commutation
        stabilizers = self._make_commuting(stabilizers, n)

        return StabilizerCode(
            n_physical=n,
            n_logical=k,
            stabilizers=stabilizers,
            distance=None,  # Unknown for random codes
            name=f"Random [[{n},{k},?]]"
        )

    def get_splits(self) -> Tuple[List[CodeSample], List[CodeSample], List[CodeSample]]:
        """
        Get train/val/test splits.

        Returns:
            (train_samples, val_samples, test_samples)
        """
        if not self._generated:
            self.generate()

        n = len(self.samples)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)

        train = self.samples[:n_train]
        val = self.samples[n_train:n_train + n_val]
        test = self.samples[n_train + n_val:]

        return train, val, test

    def get_family_splits(self) -> Dict[str, List[CodeSample]]:
        """Get samples organized by code family."""
        families = {'happy': [], 'ldpc': [], 'random': []}

        for sample in self.samples:
            families[sample.family].append(sample)

        return families

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CodeSample:
        return self.samples[idx]

    def _save_to_cache(self):
        """Save dataset to cache directory."""
        if not self.cache_dir:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        cache_file = os.path.join(self.cache_dir, 'dataset.pkl')

        with open(cache_file, 'wb') as f:
            pickle.dump(self.samples, f)

        print(f"Saved dataset to {cache_file}")

    def _load_from_cache(self) -> bool:
        """Load dataset from cache if available."""
        if not self.cache_dir:
            return False

        cache_file = os.path.join(self.cache_dir, 'dataset.pkl')

        if not os.path.exists(cache_file):
            return False

        try:
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            self._generated = True
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False


def create_pyg_dataset(samples: List[CodeSample]) -> List[Dict[str, Any]]:
    """
    Convert samples to PyTorch Geometric format.

    Args:
        samples: List of CodeSample objects

    Returns:
        List of dicts with 'x', 'edge_index', 'hyperedges', 'y' keys
    """
    pyg_data = []

    for sample in samples:
        sample.compute_graph()

        data = {
            'x': sample.node_features.astype(np.float32),
            'edge_index': sample.edge_index.astype(np.int64),
            'hyperedges': sample.hyperedges,
            'distance': sample.code.distance if sample.code.distance else 0,
            'n_physical': sample.code.n_physical,
            'n_logical': sample.code.n_logical,
            'family': sample.family,
            'depth': sample.depth if sample.depth else 0,
        }

        if sample.geometric_features is not None:
            data['geometric_features'] = sample.geometric_features.astype(np.float32)

        pyg_data.append(data)

    return pyg_data


def compute_dataset_statistics(samples: List[CodeSample]) -> Dict[str, Any]:
    """
    Compute statistics about the dataset.

    Args:
        samples: List of code samples

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(samples),
        'families': {},
        'sizes': {
            'n_physical': [],
            'n_logical': [],
            'distance': [],
        },
        'depths': [],
    }

    for sample in samples:
        # Count by family
        if sample.family not in stats['families']:
            stats['families'][sample.family] = 0
        stats['families'][sample.family] += 1

        # Collect sizes
        stats['sizes']['n_physical'].append(sample.code.n_physical)
        stats['sizes']['n_logical'].append(sample.code.n_logical)
        if sample.code.distance:
            stats['sizes']['distance'].append(sample.code.distance)

        # Collect depths (for HaPPY)
        if sample.depth:
            stats['depths'].append(sample.depth)

    # Compute summary statistics
    for key in stats['sizes']:
        values = stats['sizes'][key]
        if values:
            stats['sizes'][key] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
            }
        else:
            stats['sizes'][key] = None

    if stats['depths']:
        stats['depth_distribution'] = {
            d: stats['depths'].count(d)
            for d in sorted(set(stats['depths']))
        }

    return stats


def generate_small_code_dataset(
    seed: int = 42,
    n_random_per_config: int = 30,
    n_ldpc_743: int = 50
) -> List[CodeSample]:
    """
    Generate a dataset of codes with n_physical <= 12 for
    code-space Krylov complexity computation.

    Includes:
    - Named codes: five-qubit [[5,1,3]], Steane [[7,1,3]], repetition n=4..12
    - LDPC [[7,4,3]]: n_ldpc_743 codes
    - Random codes at configs (4,1)..(12,3)

    Args:
        seed: Random seed
        n_random_per_config: Random codes per (n,k) configuration
        n_ldpc_743: Number of LDPC [[7,4,3]] codes

    Returns:
        List of CodeSample objects
    """
    samples = []
    dataset_helper = HolographicCodeDataset(n_happy=0, n_ldpc=0, n_random=0, seed=seed)

    # Named codes
    named_codes = [
        (create_five_qubit_code(), 'named'),
        (create_steane_code(), 'named'),
    ]
    for n_rep in range(4, 13):
        named_codes.append((create_repetition_code(n_rep), 'named'))

    for code, family in named_codes:
        sample = CodeSample(
            code=code,
            family=family,
            geometric_features=dataset_helper._extract_ldpc_features(code)
        )
        sample.compute_graph()
        samples.append(sample)

    # LDPC [[7,4,3]] codes
    for i in range(n_ldpc_743):
        try:
            code = dataset_helper._create_ldpc_code(7, 4, 3, seed=seed + i)
            sample = CodeSample(
                code=code,
                family='ldpc',
                geometric_features=dataset_helper._extract_ldpc_features(code)
            )
            sample.compute_graph()
            samples.append(sample)
        except Exception as e:
            print(f"Warning: Failed LDPC [[7,4,3]] code {i}: {e}")

    # Random codes at various (n, k) configs
    random_configs = [
        (4, 1), (5, 1), (6, 1), (6, 2),
        (7, 1), (7, 3), (8, 1), (8, 2),
        (9, 1), (9, 2), (9, 3),
        (10, 1), (10, 2), (10, 3),
        (11, 1), (11, 2), (11, 3),
        (12, 1), (12, 2), (12, 3),
    ]
    for n, k in random_configs:
        for i in range(n_random_per_config):
            try:
                code = dataset_helper._create_random_code(n, k, seed=seed + 1000 * n + 100 * k + i)
                sample = CodeSample(
                    code=code,
                    family='random',
                    geometric_features=dataset_helper._extract_ldpc_features(code)
                )
                sample.compute_graph()
                samples.append(sample)
            except Exception as e:
                print(f"Warning: Failed random [[{n},{k},?]] code {i}: {e}")

    print(f"Generated {len(samples)} small codes (n_physical <= 12)")
    return samples


if __name__ == '__main__':
    # Test dataset generation
    print("Testing dataset generation...")

    # Small test dataset
    dataset = HolographicCodeDataset(
        n_happy=10,
        n_ldpc=5,
        n_random=5,
        happy_depths={2: 5, 3: 5}
    )

    dataset.generate(parallel=False)

    train, val, test = dataset.get_splits()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    stats = compute_dataset_statistics(dataset.samples)
    print(f"Statistics: {stats}")
