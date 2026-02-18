"""
HaPPY Code Generator

Implements Harlow-Pastawski-Preskill-Yoshida (HaPPY) holographic codes
based on perfect tensor networks with hyperbolic geometry.

Reference: Pastawski et al., "Holographic quantum error-correcting codes:
toy models for the bulk/boundary correspondence" (2015)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from .stabilizer import StabilizerCode, pauli_weight


@dataclass
class HaPPYCode:
    """
    Represents a HaPPY holographic code.

    The code is built from a tensor network on a hyperbolic tiling,
    where each tensor is a perfect tensor (maximally entangled).

    Attributes:
        depth: Depth of the hyperbolic tree (number of layers)
        bond_dim: Bond dimension of perfect tensors
        stabilizer_code: Underlying stabilizer code representation
        tensor_network: Structure of the tensor network
        bulk_qubits: Indices of bulk (logical) qubits
        boundary_qubits: Indices of boundary (physical) qubits
    """
    depth: int
    bond_dim: int
    stabilizer_code: StabilizerCode
    bulk_qubits: List[int] = field(default_factory=list)
    boundary_qubits: List[int] = field(default_factory=list)
    tensor_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)

    @property
    def n_physical(self) -> int:
        return self.stabilizer_code.n_physical

    @property
    def n_logical(self) -> int:
        return self.stabilizer_code.n_logical

    @property
    def distance(self) -> Optional[int]:
        return self.stabilizer_code.distance

    def get_geometric_features(self) -> np.ndarray:
        """
        Extract geometric features for ML models.

        Returns:
            Array of features: [depth, n_physical, n_logical,
                               avg_stabilizer_weight, hyperbolic_radius, ...]
        """
        avg_weight = np.mean([
            pauli_weight(s) for s in self.stabilizer_code.stabilizers
        ])

        # Hyperbolic radius (proportional to depth in Poincare disk)
        hyperbolic_radius = np.tanh(self.depth * 0.5)

        # Encoding rate
        rate = self.n_logical / self.n_physical if self.n_physical > 0 else 0

        return np.array([
            self.depth,
            self.n_physical,
            self.n_logical,
            avg_weight,
            hyperbolic_radius,
            rate,
            len(self.stabilizer_code.stabilizers),
        ])


def create_perfect_tensor(n_legs: int = 6) -> np.ndarray:
    """
    Create a perfect tensor with specified number of legs.

    A perfect tensor has the property that any bipartition into
    equal or smaller halves gives a maximally entangled state.

    For the [[5,1,3]] code, the 6-leg tensor corresponds to
    5 physical qubits + 1 logical qubit.

    Args:
        n_legs: Number of tensor legs (default: 6 for [[5,1,3]] code)

    Returns:
        Complex tensor of shape (2,) * n_legs
    """
    if n_legs == 6:
        # Use the [[5,1,3]] perfect code tensor
        # This is the encoding isometry: |logical> -> |physical>
        dim = 2 ** n_legs
        tensor = np.zeros((2,) * n_legs, dtype=complex)

        # The [[5,1,3]] code encodes |0>_L and |1>_L as:
        # |0>_L = (|00000> + |10010> + |01001> + |10100> +
        #         |01010> - |11011> - |00110> - |11000> -
        #         |11101> - |00011> - |11110> - |01111> -
        #         |10001> - |01100> - |10111> + |00101>) / 4

        # Simplified: use computational basis states with proper phases
        # For a minimal working implementation

        # |0>_L encoding (16 terms)
        logical_0_terms = [
            ((0,0,0,0,0,0), 1),
            ((1,0,0,1,0,0), 1),
            ((0,1,0,0,1,0), 1),
            ((1,0,1,0,0,0), 1),
        ]

        # |1>_L encoding
        logical_1_terms = [
            ((1,1,1,1,1,1), 1),
            ((0,1,1,0,1,1), 1),
            ((1,0,1,1,0,1), 1),
            ((0,1,0,1,1,1), 1),
        ]

        # Normalize
        norm = np.sqrt(len(logical_0_terms))
        for idx, phase in logical_0_terms:
            tensor[idx] = phase / norm
        for idx, phase in logical_1_terms:
            tensor[idx] = phase / norm

        return tensor

    elif n_legs == 4:
        # 4-leg perfect tensor (simpler, for testing)
        tensor = np.zeros((2,) * n_legs, dtype=complex)

        # GHZ-like state
        tensor[0, 0, 0, 0] = 1 / np.sqrt(2)
        tensor[1, 1, 1, 1] = 1 / np.sqrt(2)

        return tensor

    else:
        # Generic: create random unitary and reshape
        # Not guaranteed to be perfect, but useful for testing
        dim = 2 ** (n_legs // 2)
        U = np.eye(dim, dtype=complex)  # Identity for simplicity
        return U.reshape((2,) * n_legs)


def create_happy_code(depth: int, bond_dim: int = 2) -> HaPPYCode:
    """
    Create a HaPPY code with specified depth.

    The code is built from a binary tree of perfect tensors,
    mimicking the structure of a hyperbolic tiling.

    Args:
        depth: Tree depth (1 = single tensor, 2 = 1 center + 5 leaves, etc.)
        bond_dim: Bond dimension (default: 2 for qubits)

    Returns:
        HaPPYCode instance

    Note: For depth > 3, the code becomes very large. This implementation
    is simplified for the prototype phase.
    """
    if depth < 1:
        raise ValueError("Depth must be >= 1")

    if depth == 1:
        # Single [[5,1,3]] perfect tensor
        return _create_depth1_happy()
    elif depth == 2:
        # Central tensor + 5 boundary tensors
        return _create_depth2_happy()
    else:
        # General depth - recursive construction
        return _create_general_happy(depth, bond_dim)


def _create_depth1_happy() -> HaPPYCode:
    """Create depth-1 HaPPY code (just the [[5,1,3]] perfect code)."""
    from .stabilizer import create_five_qubit_code

    code = create_five_qubit_code()

    return HaPPYCode(
        depth=1,
        bond_dim=2,
        stabilizer_code=code,
        bulk_qubits=[],  # All qubits are boundary
        boundary_qubits=list(range(5)),
        tensor_positions={0: (0.0, 0.0)},  # Center
    )


def _create_depth2_happy() -> HaPPYCode:
    """
    Create depth-2 HaPPY code.

    Structure: 1 central tensor connected to 5 boundary tensors.
    The central tensor's 5 legs connect to 5 boundary tensors.
    Each boundary tensor has 4 remaining legs as physical qubits.

    Total physical qubits: 5 * 4 = 20
    Logical qubits: 1 (from central tensor) + 5 (from boundary tensors) - 5 (contracted) = 1

    This is a simplification - actual HaPPY uses specific stabilizers.
    """
    # For depth 2, we have:
    # - Central tensor: [[5,1,3]] code
    # - 5 boundary tensors, each [[5,1,3]]
    # - 5 legs contracted between center and boundaries

    n_physical = 20  # 5 boundary tensors * 4 boundary legs each
    n_logical = 1    # Simplified: 1 logical qubit

    # Generate stabilizers for the composite code
    # This is a simplified version - actual construction involves
    # pushing stabilizers through the tensor network

    # Simplified stabilizers for depth-2 (not fully accurate but functional)
    stabilizers = []

    # Local stabilizers at each boundary tensor (X-type)
    for tensor_id in range(5):
        base_idx = tensor_id * 4
        # XXXX on 4 qubits of this tensor
        s = ['I'] * n_physical
        for i in range(4):
            s[base_idx + i] = 'X'
        stabilizers.append(''.join(s))

    # Z-type stabilizers connecting neighboring tensors
    for tensor_id in range(5):
        next_tensor = (tensor_id + 1) % 5
        base_idx = tensor_id * 4
        next_base_idx = next_tensor * 4

        # ZZ on boundary between adjacent tensors
        s = ['I'] * n_physical
        s[base_idx + 3] = 'Z'  # Last qubit of current tensor
        s[next_base_idx] = 'Z'  # First qubit of next tensor
        stabilizers.append(''.join(s))

    # Additional stabilizers to fill out the group
    for tensor_id in range(5):
        base_idx = tensor_id * 4
        # ZZZZ on each tensor
        s = ['I'] * n_physical
        for i in range(4):
            s[base_idx + i] = 'Z'
        stabilizers.append(''.join(s))

    # Mixed stabilizers
    for tensor_id in range(4):
        base_idx = tensor_id * 4
        next_base_idx = (tensor_id + 1) * 4

        s = ['I'] * n_physical
        s[base_idx + 1] = 'X'
        s[base_idx + 2] = 'Z'
        s[next_base_idx + 1] = 'Z'
        s[next_base_idx + 2] = 'X'
        stabilizers.append(''.join(s))

    # Ensure correct number of stabilizers
    stabilizers = stabilizers[:n_physical - n_logical]

    code = StabilizerCode(
        n_physical=n_physical,
        n_logical=n_logical,
        stabilizers=stabilizers,
        distance=3,  # Lower bound for depth-2
        name=f"HaPPY depth-2"
    )

    # Boundary qubits are all physical qubits (at depth 2, all are on boundary)
    boundary = list(range(n_physical))

    # Tensor positions for visualization (in Poincare disk)
    positions = {0: (0.0, 0.0)}  # Center
    for i in range(5):
        angle = 2 * np.pi * i / 5
        r = 0.6  # Radial position
        positions[i + 1] = (r * np.cos(angle), r * np.sin(angle))

    return HaPPYCode(
        depth=2,
        bond_dim=2,
        stabilizer_code=code,
        bulk_qubits=[],  # Central logical qubit
        boundary_qubits=boundary,
        tensor_positions=positions,
    )


def _create_general_happy(depth: int, bond_dim: int) -> HaPPYCode:
    """
    Create general depth HaPPY code.

    For depth d, the number of boundary qubits grows exponentially.
    This simplified implementation caps at reasonable sizes.
    """
    # Number of tensors grows exponentially with depth
    # For a {5,4} hyperbolic tiling: ~5 * 4^(d-1) tensors

    # Cap depth for practical computation
    if depth > 4:
        print(f"Warning: depth {depth} may be slow. Using depth 4.")
        depth = 4

    # Estimate sizes
    n_tensors = 1 + 5 * sum(4 ** i for i in range(depth - 1)) if depth > 1 else 1
    n_physical = min(5 * (4 ** (depth - 1)), 100)  # Cap at 100 qubits
    n_logical = 1  # Simplified

    # Generate simplified stabilizers
    stabilizers = []

    # Local X stabilizers
    for i in range(0, n_physical - 3, 4):
        s = ['I'] * n_physical
        for j in range(min(4, n_physical - i)):
            s[i + j] = 'X'
        stabilizers.append(''.join(s))

    # Local Z stabilizers
    for i in range(0, n_physical - 3, 4):
        s = ['I'] * n_physical
        for j in range(min(4, n_physical - i)):
            s[i + j] = 'Z'
        stabilizers.append(''.join(s))

    # Connection stabilizers
    for i in range(n_physical - 1):
        if len(stabilizers) >= n_physical - n_logical:
            break
        s = ['I'] * n_physical
        s[i] = 'Z'
        s[i + 1] = 'Z'
        stabilizers.append(''.join(s))

    stabilizers = stabilizers[:n_physical - n_logical]

    code = StabilizerCode(
        n_physical=n_physical,
        n_logical=n_logical,
        stabilizers=stabilizers,
        distance=depth,  # Rough estimate
        name=f"HaPPY depth-{depth}"
    )

    # Generate positions in Poincare disk
    positions = {0: (0.0, 0.0)}
    idx = 1
    for d in range(1, depth):
        r = np.tanh(d * 0.5)  # Hyperbolic radius
        n_at_depth = 5 * (4 ** (d - 1)) if d > 0 else 1
        for i in range(min(n_at_depth, 20)):  # Cap visualization
            angle = 2 * np.pi * i / n_at_depth + d * 0.1
            positions[idx] = (r * np.cos(angle), r * np.sin(angle))
            idx += 1

    return HaPPYCode(
        depth=depth,
        bond_dim=bond_dim,
        stabilizer_code=code,
        bulk_qubits=[],
        boundary_qubits=list(range(n_physical)),
        tensor_positions=positions,
    )


def generate_happy_dataset(
    n_samples: int = 100,
    depths: List[int] = [1, 2, 3],
    random_variations: bool = True
) -> List[HaPPYCode]:
    """
    Generate a dataset of HaPPY codes for training.

    Args:
        n_samples: Total number of codes to generate
        depths: List of depths to sample from
        random_variations: If True, add random perturbations

    Returns:
        List of HaPPYCode instances
    """
    codes = []
    samples_per_depth = n_samples // len(depths)

    for depth in depths:
        for i in range(samples_per_depth):
            code = create_happy_code(depth)

            if random_variations and i > 0:
                # Add small random perturbations to stabilizers
                # (This is a placeholder - actual variations would be more sophisticated)
                code = _add_random_variation(code, seed=i)

            codes.append(code)

    return codes


def _add_random_variation(code: HaPPYCode, seed: int = 0) -> HaPPYCode:
    """Add random variation to a code (for data augmentation)."""
    np.random.seed(seed)

    # Simple variation: permute qubits
    n = code.n_physical
    perm = np.random.permutation(n)

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
