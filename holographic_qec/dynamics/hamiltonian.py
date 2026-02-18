"""
Extended Hamiltonian Builders

Provides various quantum Hamiltonians for complexity dynamics studies:
- XXZ spin chain (with anisotropy parameter Delta)
- Transverse field Ising model
- Random local Hamiltonians
- Utility functions for parameter encoding
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from scipy import linalg


# Pauli matrices
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


@dataclass
class HamiltonianParams:
    """
    Parameters describing a Hamiltonian for encoding.

    Provides a standardized way to encode Hamiltonian properties
    for neural network input.
    """
    model_type: str  # 'xxz', 'ising', 'random'
    n_qubits: int
    coupling_strength: float = 1.0
    field_strength: float = 0.0
    anisotropy: float = 1.0  # Delta for XXZ
    periodic: bool = False
    temperature: float = 0.0  # For thermal states

    def to_vector(self) -> np.ndarray:
        """
        Convert to feature vector for neural network input.

        Returns:
            8-dimensional feature vector
        """
        # One-hot encode model type
        model_types = {'xxz': [1, 0, 0], 'ising': [0, 1, 0], 'random': [0, 0, 1]}
        model_vec = model_types.get(self.model_type, [0, 0, 0])

        return np.array([
            *model_vec,
            self.n_qubits / 10.0,  # Normalized
            self.coupling_strength,
            self.field_strength,
            self.anisotropy,
            1.0 if self.periodic else 0.0
        ], dtype=np.float32)

    @staticmethod
    def vector_dim() -> int:
        """Return dimension of parameter vector."""
        return 8


class HamiltonianBuilder:
    """
    Factory class for building various Hamiltonians.

    All methods return both the Hamiltonian matrix and the
    corresponding HamiltonianParams for neural network encoding.
    """

    @staticmethod
    def xxz(
        n_qubits: int,
        J_xy: float = 1.0,
        J_z: float = 1.0,
        h: float = 0.0,
        periodic: bool = False
    ) -> Tuple[np.ndarray, HamiltonianParams]:
        """
        Build XXZ Hamiltonian.

        H = J_xy * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y)
          + J_z * sum_i S_i^z S_{i+1}^z
          + h * sum_i S_i^z

        The anisotropy parameter is Delta = J_z / J_xy.

        Args:
            n_qubits: Number of spins
            J_xy: XY coupling strength
            J_z: Z coupling strength (anisotropy: Delta = J_z/J_xy)
            h: External magnetic field
            periodic: Use periodic boundary conditions

        Returns:
            (H, params): Hamiltonian matrix and parameters
        """
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        # Number of bonds
        n_bonds = n_qubits if periodic else n_qubits - 1

        for i in range(n_bonds):
            j = (i + 1) % n_qubits

            # XX + YY term (using ladder operators: S+S- + S-S+)
            # S_i^x S_j^x + S_i^y S_j^y = (1/2)(S_i^+ S_j^- + S_i^- S_j^+)
            H += J_xy * 0.5 * (
                _tensor_product(X, X, i, j, n_qubits) +
                _tensor_product(Y, Y, i, j, n_qubits)
            )

            # ZZ term
            H += J_z * 0.25 * _tensor_product(Z, Z, i, j, n_qubits)

        # External field
        if h != 0:
            for i in range(n_qubits):
                H += h * 0.5 * _single_site_op(Z, i, n_qubits)

        # Compute anisotropy
        Delta = J_z / J_xy if J_xy != 0 else 0.0

        params = HamiltonianParams(
            model_type='xxz',
            n_qubits=n_qubits,
            coupling_strength=J_xy,
            field_strength=h,
            anisotropy=Delta,
            periodic=periodic
        )

        return H, params

    @staticmethod
    def ising(
        n_qubits: int,
        J: float = 1.0,
        g: float = 1.0,
        periodic: bool = False
    ) -> Tuple[np.ndarray, HamiltonianParams]:
        """
        Build transverse field Ising model Hamiltonian.

        H = -J * sum_i Z_i Z_{i+1} - g * sum_i X_i

        Critical point at g/J = 1 (quantum phase transition).

        Args:
            n_qubits: Number of spins
            J: Ising coupling strength
            g: Transverse field strength
            periodic: Use periodic boundary conditions

        Returns:
            (H, params): Hamiltonian matrix and parameters
        """
        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        # Number of bonds
        n_bonds = n_qubits if periodic else n_qubits - 1

        # ZZ interactions
        for i in range(n_bonds):
            j = (i + 1) % n_qubits
            H -= J * _tensor_product(Z, Z, i, j, n_qubits)

        # Transverse field
        for i in range(n_qubits):
            H -= g * _single_site_op(X, i, n_qubits)

        params = HamiltonianParams(
            model_type='ising',
            n_qubits=n_qubits,
            coupling_strength=J,
            field_strength=g,
            anisotropy=g / J if J != 0 else 0.0,  # g/J ratio
            periodic=periodic
        )

        return H, params

    @staticmethod
    def random_local(
        n_qubits: int,
        locality: int = 2,
        strength: float = 1.0,
        seed: Optional[int] = None,
        field_strength: float = 0.0
    ) -> Tuple[np.ndarray, HamiltonianParams]:
        """
        Build random k-local Hamiltonian.

        Generates a Hamiltonian as a sum of random k-local terms,
        where each term acts on k neighboring qubits.

        Args:
            n_qubits: Number of qubits
            locality: Number of qubits each term acts on
            strength: Overall coupling strength
            seed: Random seed for reproducibility
            field_strength: External field strength (Z-field on each qubit)

        Returns:
            (H, params): Hamiltonian matrix and parameters
        """
        if seed is not None:
            np.random.seed(seed)

        dim = 2 ** n_qubits
        H = np.zeros((dim, dim), dtype=complex)

        paulis = [I2, X, Y, Z]

        # Add random local terms
        for i in range(n_qubits - locality + 1):
            # Random Pauli string of length `locality`
            pauli_indices = np.random.randint(1, 4, size=locality)  # 1-3 for X,Y,Z
            coefficient = np.random.randn() * strength

            # Build the local operator
            local_op = np.eye(1)
            for j in range(n_qubits):
                if i <= j < i + locality:
                    local_op = np.kron(local_op, paulis[pauli_indices[j - i]])
                else:
                    local_op = np.kron(local_op, I2)

            H += coefficient * local_op

        # Ensure Hermiticity
        H = (H + H.conj().T) / 2

        # Add external Z-field if non-zero
        if field_strength != 0:
            for i in range(n_qubits):
                H += field_strength * _single_site_op(Z, i, n_qubits)

        params = HamiltonianParams(
            model_type='random',
            n_qubits=n_qubits,
            coupling_strength=strength,
            field_strength=field_strength,
            anisotropy=locality / n_qubits,  # Encode locality ratio
            periodic=False
        )

        return H, params

    @staticmethod
    def heisenberg(
        n_qubits: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = False
    ) -> Tuple[np.ndarray, HamiltonianParams]:
        """
        Build isotropic Heisenberg (XXX) Hamiltonian.

        H = J * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + S_i^z S_{i+1}^z)
          + h * sum_i S_i^z

        This is the XXZ model with Delta = 1.

        Args:
            n_qubits: Number of spins
            J: Exchange coupling
            h: External field
            periodic: Periodic boundaries

        Returns:
            (H, params): Hamiltonian matrix and parameters
        """
        return HamiltonianBuilder.xxz(n_qubits, J, J, h, periodic)


def _single_site_op(op: np.ndarray, site: int, n_qubits: int) -> np.ndarray:
    """Build n-qubit operator with `op` on given site, identity elsewhere."""
    result = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        if i == site:
            result = np.kron(result, op)
        else:
            result = np.kron(result, I2)
    return result


def _tensor_product(
    op1: np.ndarray,
    op2: np.ndarray,
    site1: int,
    site2: int,
    n_qubits: int
) -> np.ndarray:
    """Build n-qubit operator with op1 on site1, op2 on site2."""
    result = np.eye(1, dtype=complex)
    for i in range(n_qubits):
        if i == site1:
            result = np.kron(result, op1)
        elif i == site2:
            result = np.kron(result, op2)
        else:
            result = np.kron(result, I2)
    return result


def compute_hamiltonian_spectrum(H: np.ndarray) -> Dict[str, Any]:
    """
    Compute spectral properties of a Hamiltonian.

    Returns:
        Dict with eigenvalues, gaps, and other spectral info
    """
    eigenvalues = linalg.eigvalsh(H)
    eigenvalues = np.sort(eigenvalues)

    ground_state_energy = eigenvalues[0]
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
    bandwidth = eigenvalues[-1] - eigenvalues[0]

    return {
        'eigenvalues': eigenvalues,
        'ground_state_energy': ground_state_energy,
        'spectral_gap': spectral_gap,
        'bandwidth': bandwidth,
        'mean_energy': np.mean(eigenvalues),
        'energy_variance': np.var(eigenvalues)
    }


def generate_hamiltonian_dataset(
    xxz_deltas: List[float] = [0.0, 0.5, 1.0, 2.0],
    ising_hs: List[float] = [0.0, 0.5, 1.0, 2.0],
    n_qubits_list: List[int] = [4, 5, 6],
    n_random: int = 100,
    xxz_fields: List[float] = [0.0]
) -> List[Tuple[np.ndarray, HamiltonianParams]]:
    """
    Generate a dataset of Hamiltonians for training.

    Following the strategy document:
    - XXZ with Delta in {0.0, 0.5, 1.0, 2.0}
    - Ising with h in {0.0, 0.5, 1.0, 2.0}
    - Random local Hamiltonians for variety

    Args:
        xxz_deltas: List of anisotropy values for XXZ
        ising_hs: List of transverse field values for Ising
        n_qubits_list: System sizes to sample
        n_random: Number of random Hamiltonians
        xxz_fields: List of external field values for XXZ models

    Returns:
        List of (Hamiltonian, params) tuples
    """
    dataset = []

    # XXZ variants with field sweep
    for n_qubits in n_qubits_list:
        for delta in xxz_deltas:
            for h in xxz_fields:
                J_xy = 1.0
                J_z = delta * J_xy  # Delta = J_z / J_xy
                H, params = HamiltonianBuilder.xxz(n_qubits, J_xy, J_z, h=h)
                dataset.append((H, params))

    # Ising variants
    for n_qubits in n_qubits_list:
        for h in ising_hs:
            H, params = HamiltonianBuilder.ising(n_qubits, J=1.0, g=h)
            dataset.append((H, params))

    # Random Hamiltonians with varying field strength
    field_choices = [0.0, 0.5, 1.0, 1.5]
    for i in range(n_random):
        n = np.random.choice(n_qubits_list)
        field = field_choices[i % len(field_choices)]
        H, params = HamiltonianBuilder.random_local(n, locality=2, seed=i, field_strength=field)
        dataset.append((H, params))

    return dataset


def create_initial_states(
    n_qubits: int,
    state_type: str = 'random'
) -> np.ndarray:
    """
    Create initial states for time evolution.

    Args:
        n_qubits: Number of qubits
        state_type: 'random', 'computational', 'product', 'neel'

    Returns:
        Initial state vector (normalized)
    """
    dim = 2 ** n_qubits

    if state_type == 'random':
        # Random complex state
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)

    elif state_type == 'computational':
        # Random computational basis state
        idx = np.random.randint(0, dim)
        psi = np.zeros(dim, dtype=complex)
        psi[idx] = 1.0

    elif state_type == 'product':
        # Random product state
        psi = np.array([1], dtype=complex)
        for _ in range(n_qubits):
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            qubit = np.array([np.cos(theta/2), np.exp(1j * phi) * np.sin(theta/2)])
            psi = np.kron(psi, qubit)

    elif state_type == 'neel':
        # Neel (antiferromagnetic) state: |010101...>
        idx = sum(2 ** i for i in range(0, n_qubits, 2))
        psi = np.zeros(dim, dtype=complex)
        psi[idx] = 1.0

    else:
        raise ValueError(f"Unknown state type: {state_type}")

    # Normalize
    psi = psi / np.linalg.norm(psi)

    return psi


if __name__ == '__main__':
    print("Testing Hamiltonian builders...")

    # Test XXZ
    H_xxz, params_xxz = HamiltonianBuilder.xxz(4, J_xy=1.0, J_z=0.5)
    print(f"XXZ(4): shape={H_xxz.shape}, params={params_xxz.to_vector()}")

    # Test Ising
    H_ising, params_ising = HamiltonianBuilder.ising(4, J=1.0, g=1.0)
    print(f"Ising(4): shape={H_ising.shape}, params={params_ising.to_vector()}")

    # Test spectrum
    spec = compute_hamiltonian_spectrum(H_xxz)
    print(f"XXZ spectrum: gap={spec['spectral_gap']:.4f}, bandwidth={spec['bandwidth']:.4f}")

    # Test dataset generation
    dataset = generate_hamiltonian_dataset(
        xxz_deltas=[0.0, 1.0],
        ising_hs=[0.0, 1.0],
        n_qubits_list=[4, 5],
        n_random=10
    )
    print(f"Generated {len(dataset)} Hamiltonians")

    # Test initial states
    psi = create_initial_states(4, 'random')
    print(f"Random state: norm={np.linalg.norm(psi):.6f}")
