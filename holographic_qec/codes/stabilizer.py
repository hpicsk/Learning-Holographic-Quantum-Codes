"""
Stabilizer Code Utilities

Provides basic data structures and utilities for working with
stabilizer quantum error correcting codes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
import numpy as np
from itertools import product


# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULIS = {'I': I, 'X': X, 'Y': Y, 'Z': Z}


def pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """
    Convert a Pauli string like 'XZZXI' to its 2^n x 2^n matrix representation.

    Args:
        pauli_str: String of I, X, Y, Z characters (length n)

    Returns:
        2^n x 2^n complex matrix
    """
    if not pauli_str:
        raise ValueError("Empty Pauli string")
    mat = PAULIS[pauli_str[0]]
    for c in pauli_str[1:]:
        mat = np.kron(mat, PAULIS[c])
    return mat


def create_codespace_state(code: 'StabilizerCode', seed: Optional[int] = None) -> np.ndarray:
    """
    Create a normalized state in the code space by projecting onto the
    +1 eigenspace of all stabilizers.

    Projects a starting state via P = prod_i (I + g_i) / 2 for each
    stabilizer generator g_i, normalizing after each projection.

    Args:
        code: A StabilizerCode with n_physical <= 12
        seed: If provided, use a Haar-random starting state with this seed
              instead of computational basis states. Different seeds yield
              different states within the code space (for k > 0 codes).

    Returns:
        Normalized state vector in the code space (length 2^n_physical)

    Raises:
        ValueError: If n_physical > 12 or all starting states project to zero
    """
    n = code.n_physical
    if n > 12:
        raise ValueError(f"n_physical={n} too large for matrix exponentiation (max 12)")

    dim = 2 ** n
    identity = np.eye(dim, dtype=complex)

    # Precompute stabilizer matrices
    stab_matrices = [pauli_string_to_matrix(s) for s in code.stabilizers]

    def _try_project(psi: np.ndarray) -> Optional[np.ndarray]:
        state = psi.copy()
        for g_mat in stab_matrices:
            projector = (identity + g_mat) / 2.0
            state = projector @ state
            norm = np.linalg.norm(state)
            if norm < 1e-12:
                return None
            state = state / norm
        return state

    # If seed is given, use Haar-random starting state
    if seed is not None:
        rng = np.random.RandomState(seed)
        psi_rand = rng.randn(dim) + 1j * rng.randn(dim)
        psi_rand = psi_rand / np.linalg.norm(psi_rand)
        result = _try_project(psi_rand)
        if result is not None:
            return result
        raise ValueError(f"Haar-random state (seed={seed}) projects to zero for code {code.name}")

    # Default: try computational basis states |0...0>, |10...0>, |010...0>, etc.
    candidates = list(range(min(n + 1, dim)))

    for idx in candidates:
        psi0 = np.zeros(dim, dtype=complex)
        psi0[idx] = 1.0
        result = _try_project(psi0)
        if result is not None:
            return result

    # Random state fallback
    rng = np.random.RandomState(42)
    psi_rand = rng.randn(dim) + 1j * rng.randn(dim)
    psi_rand = psi_rand / np.linalg.norm(psi_rand)
    result = _try_project(psi_rand)
    if result is not None:
        return result

    raise ValueError(f"All starting states project to zero for code {code.name}")


@dataclass
class StabilizerCode:
    """
    Represents a stabilizer quantum error correcting code.

    Attributes:
        n_physical: Number of physical qubits
        n_logical: Number of logical qubits (k)
        stabilizers: List of stabilizer generators as Pauli strings
        logical_x: Logical X operators
        logical_z: Logical Z operators
        distance: Code distance (None if not computed)
        name: Optional name for the code
    """
    n_physical: int
    n_logical: int
    stabilizers: List[str]
    logical_x: List[str] = field(default_factory=list)
    logical_z: List[str] = field(default_factory=list)
    distance: Optional[int] = None
    name: str = "unnamed"

    def __post_init__(self):
        """Validate the code parameters."""
        # Check stabilizer count
        expected_stabilizers = self.n_physical - self.n_logical
        if len(self.stabilizers) != expected_stabilizers:
            raise ValueError(
                f"Expected {expected_stabilizers} stabilizers for [[{self.n_physical},{self.n_logical}]] code, "
                f"got {len(self.stabilizers)}"
            )

        # Validate stabilizer strings
        for s in self.stabilizers:
            if len(s) != self.n_physical:
                raise ValueError(f"Stabilizer '{s}' has wrong length")
            if not all(c in 'IXYZ' for c in s):
                raise ValueError(f"Invalid Pauli character in '{s}'")

    @property
    def code_parameters(self) -> str:
        """Return [[n, k, d]] notation."""
        d = self.distance if self.distance else '?'
        return f"[[{self.n_physical}, {self.n_logical}, {d}]]"

    def get_stabilizer_matrix(self) -> np.ndarray:
        """
        Convert stabilizers to binary symplectic form.

        Returns:
            H: (n-k) x 2n matrix where H = [Hx | Hz]
               Hx[i,j] = 1 if stabilizer i has X or Y on qubit j
               Hz[i,j] = 1 if stabilizer i has Z or Y on qubit j
        """
        n = self.n_physical
        m = len(self.stabilizers)
        H = np.zeros((m, 2*n), dtype=int)

        for i, stab in enumerate(self.stabilizers):
            for j, p in enumerate(stab):
                if p in ('X', 'Y'):
                    H[i, j] = 1
                if p in ('Z', 'Y'):
                    H[i, n + j] = 1

        return H

    def to_graph(self) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
        """
        Convert code to graph representation for GNN.

        Returns:
            node_features: (n_physical, 4) array
                - qubit index (normalized)
                - degree in stabilizer graph
                - number of X/Y in connected stabilizers
                - number of Z/Y in connected stabilizers
            edge_index: (2, num_edges) array for qubit-qubit connections
            hyperedges: List of qubit indices for each stabilizer
        """
        n = self.n_physical

        # Build hyperedges (which qubits each stabilizer acts on)
        hyperedges = []
        for stab in self.stabilizers:
            qubits = [i for i, p in enumerate(stab) if p != 'I']
            hyperedges.append(qubits)

        # Compute node features
        node_features = np.zeros((n, 4))
        for i in range(n):
            node_features[i, 0] = i / n  # normalized index

            # Count stabilizers acting on this qubit
            degree = 0
            x_count = 0
            z_count = 0
            for stab in self.stabilizers:
                if stab[i] != 'I':
                    degree += 1
                    if stab[i] in ('X', 'Y'):
                        x_count += 1
                    if stab[i] in ('Z', 'Y'):
                        z_count += 1

            node_features[i, 1] = degree / len(self.stabilizers)
            node_features[i, 2] = x_count / max(1, degree)
            node_features[i, 3] = z_count / max(1, degree)

        # Build edge index (qubits connected if they share a stabilizer)
        edges = set()
        for qubits in hyperedges:
            for i in range(len(qubits)):
                for j in range(i + 1, len(qubits)):
                    edges.add((qubits[i], qubits[j]))
                    edges.add((qubits[j], qubits[i]))  # undirected

        if edges:
            edge_index = np.array(list(edges)).T
        else:
            edge_index = np.zeros((2, 0), dtype=int)

        return node_features, edge_index, hyperedges


def pauli_weight(pauli_string: str) -> int:
    """
    Compute the weight (number of non-identity operators) of a Pauli string.

    Args:
        pauli_string: String of I, X, Y, Z characters

    Returns:
        Number of non-I characters
    """
    return sum(1 for p in pauli_string if p != 'I')


def pauli_product(p1: str, p2: str) -> Tuple[str, complex]:
    """
    Compute product of two Pauli strings.

    Args:
        p1, p2: Pauli strings of same length

    Returns:
        (result_string, phase) where phase is in {1, -1, 1j, -1j}
    """
    if len(p1) != len(p2):
        raise ValueError("Pauli strings must have same length")

    result = []
    total_phase = 1

    # Single-qubit Pauli multiplication table (with phases)
    mult_table = {
        ('I', 'I'): ('I', 1), ('I', 'X'): ('X', 1), ('I', 'Y'): ('Y', 1), ('I', 'Z'): ('Z', 1),
        ('X', 'I'): ('X', 1), ('X', 'X'): ('I', 1), ('X', 'Y'): ('Z', 1j), ('X', 'Z'): ('Y', -1j),
        ('Y', 'I'): ('Y', 1), ('Y', 'X'): ('Z', -1j), ('Y', 'Y'): ('I', 1), ('Y', 'Z'): ('X', 1j),
        ('Z', 'I'): ('Z', 1), ('Z', 'X'): ('Y', 1j), ('Z', 'Y'): ('X', -1j), ('Z', 'Z'): ('I', 1),
    }

    for a, b in zip(p1, p2):
        r, phase = mult_table[(a, b)]
        result.append(r)
        total_phase *= phase

    return ''.join(result), total_phase


def commutes(p1: str, p2: str) -> bool:
    """
    Check if two Pauli strings commute.

    Two Paulis commute if they have an even number of positions
    where one is X/Y and the other is Z/Y.
    """
    if len(p1) != len(p2):
        raise ValueError("Pauli strings must have same length")

    anticommute_count = 0
    for a, b in zip(p1, p2):
        if a != 'I' and b != 'I' and a != b:
            # Both non-identity and different -> anticommute at this position
            # (except I always commutes)
            if (a in ('X', 'Y') and b in ('Z', 'Y')) or (a in ('Z', 'Y') and b in ('X', 'Y')):
                if a != b:  # X-Z or Z-X or X-Y or Y-X or Y-Z or Z-Y
                    anticommute_count += 1

    return anticommute_count % 2 == 0


def compute_code_distance(code: StabilizerCode, max_weight: int = None) -> int:
    """
    Compute the code distance by brute force enumeration.

    The distance is the minimum weight of a logical operator,
    i.e., a Pauli that commutes with all stabilizers but is not
    in the stabilizer group.

    Args:
        code: The stabilizer code
        max_weight: Maximum weight to search (default: n)

    Returns:
        Code distance d

    Warning: Exponential in max_weight! Only use for small codes.
    """
    n = code.n_physical
    if max_weight is None:
        max_weight = n

    # Generate stabilizer group (up to reasonable size)
    stabilizer_group = set()
    stabilizer_group.add('I' * n)  # Identity is always in

    # Add generators and their products (simplified - just generators for small codes)
    for s in code.stabilizers:
        stabilizer_group.add(s)

    # Generate products of stabilizers (limited depth)
    current = set(code.stabilizers)
    for _ in range(min(4, len(code.stabilizers))):  # Limit iterations
        new_elements = set()
        for s1 in current:
            for s2 in code.stabilizers:
                prod, _ = pauli_product(s1, s2)
                if prod not in stabilizer_group:
                    new_elements.add(prod)
                    stabilizer_group.add(prod)
        current = new_elements
        if not current:
            break

    # Search for minimum weight logical operator
    for weight in range(1, max_weight + 1):
        # Enumerate all Paulis of given weight
        for positions in product(range(n), repeat=weight):
            if len(set(positions)) != weight:  # Skip if positions repeat
                continue

            for paulis in product('XYZ', repeat=weight):
                # Build Pauli string
                pauli = ['I'] * n
                for pos, p in zip(positions, paulis):
                    pauli[pos] = p
                pauli_str = ''.join(pauli)

                # Check if it commutes with all stabilizers
                commutes_with_all = all(
                    commutes(pauli_str, s) for s in code.stabilizers
                )

                if commutes_with_all and pauli_str not in stabilizer_group:
                    # Found a logical operator!
                    return weight

    return max_weight + 1  # Distance exceeds search limit


def create_repetition_code(n: int) -> StabilizerCode:
    """
    Create the n-qubit repetition code.

    [[n, 1, n]] code that encodes one logical qubit.
    Stabilizers: Z_i Z_{i+1} for i = 0, ..., n-2
    """
    stabilizers = []
    for i in range(n - 1):
        s = ['I'] * n
        s[i] = 'Z'
        s[i + 1] = 'Z'
        stabilizers.append(''.join(s))

    # Logical operators
    logical_x = ['X' * n]  # X on all qubits
    logical_z = ['Z' + 'I' * (n - 1)]  # Z on first qubit

    return StabilizerCode(
        n_physical=n,
        n_logical=1,
        stabilizers=stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
        distance=n,
        name=f"{n}-qubit repetition"
    )


def create_steane_code() -> StabilizerCode:
    """
    Create the [[7, 1, 3]] Steane code.

    The smallest code that can correct arbitrary single-qubit errors.
    """
    stabilizers = [
        'IIIXXXX',  # X stabilizers
        'IXXIIXX',
        'XIXIXIX',
        'IIIZZZZ',  # Z stabilizers
        'IZZIIZZ',
        'ZIZIZIZ',
    ]

    logical_x = ['XXXXXXX']
    logical_z = ['ZZZZZZZ']

    return StabilizerCode(
        n_physical=7,
        n_logical=1,
        stabilizers=stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
        distance=3,
        name="Steane [[7,1,3]]"
    )


def create_five_qubit_code() -> StabilizerCode:
    """
    Create the [[5, 1, 3]] perfect code.

    The smallest code that can correct arbitrary single-qubit errors.
    Also known as the "perfect" code because it saturates quantum
    Hamming bound.
    """
    stabilizers = [
        'XZZXI',
        'IXZZX',
        'XIXZZ',
        'ZXIXZ',
    ]

    logical_x = ['XXXXX']
    logical_z = ['ZZZZZ']

    return StabilizerCode(
        n_physical=5,
        n_logical=1,
        stabilizers=stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
        distance=3,
        name="Perfect [[5,1,3]]"
    )
