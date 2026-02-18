"""
Krylov Complexity Computation

Implements the Lanczos algorithm for computing Krylov subspace
and spread complexity, which measures operator growth under
Hamiltonian time evolution.

Reference: Parker et al., "A Universal Operator Growth Hypothesis" (2019)
"""

import numpy as np
from scipy import linalg
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class KrylovResult:
    """
    Results from Krylov complexity computation.

    Attributes:
        times: Array of time points
        complexity: C_K(t) values
        lanczos_coeffs: (alpha, beta) coefficients from Lanczos
        krylov_dimension: Dimension of Krylov subspace
        growth_exponent: Fitted early-time exponent (C_K ~ t^alpha)
        saturation_value: Long-time saturation value
    """
    times: np.ndarray
    complexity: np.ndarray
    lanczos_coeffs: Tuple[np.ndarray, np.ndarray]
    krylov_dimension: int
    growth_exponent: Optional[float] = None
    saturation_value: Optional[float] = None


def compute_krylov_basis(
    H: np.ndarray,
    psi0: np.ndarray,
    max_dim: int = 50,
    tol: float = 1e-10
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Compute Krylov basis using Lanczos algorithm.

    The Krylov subspace is span{psi0, H*psi0, H^2*psi0, ...}.
    Lanczos orthogonalizes this into an orthonormal basis with
    tridiagonal representation.

    Args:
        H: Hamiltonian matrix (d x d), must be Hermitian
        psi0: Initial state vector (d,)
        max_dim: Maximum Krylov dimension
        tol: Tolerance for detecting zero beta (subspace termination)

    Returns:
        K: List of Krylov basis vectors
        alpha: Diagonal elements (Lanczos coefficients), length = len(K)
        beta: Off-diagonal elements (Lanczos coefficients), length = len(K) - 1
    """
    d = len(psi0)
    max_dim = min(max_dim, d)

    # Normalize initial state
    psi0 = psi0 / np.linalg.norm(psi0)

    K = [psi0.copy()]
    alpha = []
    beta = []

    # Compute alpha_0 for the first basis vector
    w = H @ K[0]
    alpha_0 = np.real(np.vdot(K[0], w))
    alpha.append(alpha_0)

    # Orthogonalize
    w = w - alpha_0 * K[0]

    for j in range(max_dim - 1):
        # Full reorthogonalization (for numerical stability)
        for k in range(len(K)):
            w = w - np.vdot(K[k], w) * K[k]

        # Compute off-diagonal element
        beta_j = np.linalg.norm(w)

        if beta_j < tol:
            # Krylov subspace terminates (invariant subspace found)
            break

        # Normalize and store new basis vector
        K.append(w / beta_j)
        beta.append(beta_j)

        # Apply Hamiltonian to new vector
        w = H @ K[-1]

        # Compute diagonal element for new vector
        alpha_j = np.real(np.vdot(K[-1], w))
        alpha.append(alpha_j)

        # Orthogonalize against last two vectors (three-term recurrence)
        w = w - alpha_j * K[-1] - beta_j * K[-2]

    return K, np.array(alpha), np.array(beta)


def compute_krylov_complexity(
    H: np.ndarray,
    psi0: np.ndarray,
    t_max: float,
    n_steps: int = 100,
    max_krylov_dim: int = 50
) -> KrylovResult:
    """
    Compute Krylov complexity C_K(t) for time evolution.

    C_K(t) measures how spread the time-evolved state is in
    the Krylov basis. It's defined as:

    C_K(t) = sum_n n * |<K_n|psi(t)>|^2

    where psi(t) = exp(-iHt)|psi0> and {K_n} is the Krylov basis.

    Args:
        H: Hamiltonian matrix (d x d)
        psi0: Initial state vector (d,)
        t_max: Maximum evolution time
        n_steps: Number of time points

    Returns:
        KrylovResult with complexity curve and analysis
    """
    # Compute Krylov basis
    K, alpha, beta = compute_krylov_basis(H, psi0, max_krylov_dim)
    krylov_dim = len(K)

    if krylov_dim == 0 or len(alpha) == 0:
        # Edge case: trivial state
        times = np.linspace(0, t_max, n_steps)
        return KrylovResult(
            times=times,
            complexity=np.zeros(n_steps),
            lanczos_coeffs=(alpha, beta),
            krylov_dimension=0
        )

    # The tridiagonal matrix dimension is len(alpha) = len(K)
    # alpha has d elements (diagonal), beta has d-1 elements (off-diagonal)
    tri_dim = len(alpha)

    # Build tridiagonal matrix in Krylov basis
    T = np.diag(alpha)
    if len(beta) > 0:
        T = T + np.diag(beta, 1) + np.diag(beta, -1)

    # Time evolution in Krylov basis
    times = np.linspace(0, t_max, n_steps)
    complexity = np.zeros(n_steps)

    # Initial state in Krylov basis is |0> = (1, 0, 0, ...)
    e_0 = np.zeros(tri_dim, dtype=complex)
    e_0[0] = 1.0

    for i, t in enumerate(times):
        # Time evolution: |psi(t)> = exp(-iT*t)|0>
        if t == 0:
            c_t = e_0.copy()
        else:
            U_t = linalg.expm(-1j * T * t)
            c_t = U_t @ e_0

        # Compute complexity: C_K = sum_n n * |c_n|^2
        probs = np.abs(c_t) ** 2
        complexity[i] = np.sum(np.arange(tri_dim) * probs)

    # Extract scaling exponent
    growth_exp = extract_scaling_exponent(times, complexity)

    # Find saturation value
    saturation = np.mean(complexity[-n_steps//5:]) if n_steps > 5 else complexity[-1]

    return KrylovResult(
        times=times,
        complexity=complexity,
        lanczos_coeffs=(alpha, beta),
        krylov_dimension=krylov_dim,
        growth_exponent=growth_exp,
        saturation_value=saturation
    )


def extract_scaling_exponent(
    times: np.ndarray,
    complexity: np.ndarray,
    t_min_frac: float = 0.05,
    t_max_frac: float = 0.3
) -> Optional[float]:
    """
    Extract early-time scaling exponent from C_K(t) ~ t^alpha.

    Args:
        times: Time array
        complexity: Complexity array
        t_min_frac: Start of fit window (fraction of t_max)
        t_max_frac: End of fit window (fraction of t_max)

    Returns:
        Scaling exponent alpha, or None if fit fails
    """
    t_max = times[-1]
    t_min = t_max * t_min_frac
    t_fit_max = t_max * t_max_frac

    # Select fitting window
    mask = (times > t_min) & (times < t_fit_max) & (complexity > 1e-10)

    if np.sum(mask) < 5:
        return None

    t_fit = times[mask]
    c_fit = complexity[mask]

    # Fit log(C_K) = alpha * log(t) + const
    try:
        log_t = np.log(t_fit)
        log_c = np.log(c_fit)
        coeffs = np.polyfit(log_t, log_c, 1)
        return coeffs[0]  # Slope = exponent
    except:
        return None


def build_xxz_hamiltonian(
    n_qubits: int,
    J_xy: float = 1.0,
    J_z: float = 1.0,
    h: float = 0.0,
    periodic: bool = False
) -> np.ndarray:
    """
    Build XXZ Hamiltonian for a spin chain.

    H = J_xy * sum_i (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y)
        + J_z * sum_i S_i^z S_{i+1}^z
        + h * sum_i S_i^z

    Args:
        n_qubits: Number of spins
        J_xy: XY coupling strength
        J_z: Z coupling strength
        h: External field strength
        periodic: Use periodic boundary conditions

    Returns:
        Hamiltonian matrix (2^n x 2^n)
    """
    dim = 2 ** n_qubits

    # Pauli matrices
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    def tensor_op(op1, op2, i, j, n):
        """Build n-qubit operator with op1 on site i, op2 on site j."""
        ops = [I] * n
        ops[i] = op1
        if i != j:
            ops[j] = op2
        result = ops[0]
        for k in range(1, n):
            result = np.kron(result, ops[k])
        return result

    # Interaction terms
    n_bonds = n_qubits if periodic else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits

        # XX + YY term
        H += J_xy * 0.5 * (tensor_op(X, X, i, j, n_qubits) +
                          tensor_op(Y, Y, i, j, n_qubits))

        # ZZ term
        H += J_z * 0.25 * tensor_op(Z, Z, i, j, n_qubits)

    # External field
    if h != 0:
        for i in range(n_qubits):
            H += h * 0.5 * tensor_op(Z, I, i, i, n_qubits)

    return H


def build_ising_hamiltonian(
    n_qubits: int,
    J: float = 1.0,
    g: float = 1.0,
    periodic: bool = False
) -> np.ndarray:
    """
    Build transverse field Ising model Hamiltonian.

    H = -J * sum_i Z_i Z_{i+1} - g * sum_i X_i

    Args:
        n_qubits: Number of spins
        J: Ising coupling strength
        g: Transverse field strength
        periodic: Use periodic boundary conditions

    Returns:
        Hamiltonian matrix (2^n x 2^n)
    """
    dim = 2 ** n_qubits

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    def single_op(op, site, n):
        """Build n-qubit operator with op on given site."""
        ops = [I] * n
        ops[site] = op
        result = ops[0]
        for k in range(1, n):
            result = np.kron(result, ops[k])
        return result

    def two_op(op1, op2, i, j, n):
        """Build two-site operator."""
        ops = [I] * n
        ops[i] = op1
        ops[j] = op2
        result = ops[0]
        for k in range(1, n):
            result = np.kron(result, ops[k])
        return result

    # ZZ interactions
    n_bonds = n_qubits if periodic else n_qubits - 1
    for i in range(n_bonds):
        j = (i + 1) % n_qubits
        H -= J * two_op(Z, Z, i, j, n_qubits)

    # Transverse field
    for i in range(n_qubits):
        H -= g * single_op(X, i, n_qubits)

    return H


def compute_complexity_features(result: KrylovResult) -> np.ndarray:
    """
    Extract features from Krylov complexity computation for ML.

    Args:
        result: KrylovResult from compute_krylov_complexity

    Returns:
        Feature array: [krylov_dim, growth_exp, saturation,
                       avg_alpha, avg_beta, max_complexity, ...]
    """
    alpha, beta = result.lanczos_coeffs

    features = [
        result.krylov_dimension,
        result.growth_exponent if result.growth_exponent else 0.0,
        result.saturation_value if result.saturation_value else 0.0,
        np.mean(alpha) if len(alpha) > 0 else 0.0,
        np.std(alpha) if len(alpha) > 0 else 0.0,
        np.mean(beta) if len(beta) > 0 else 0.0,
        np.std(beta) if len(beta) > 0 else 0.0,
        np.max(result.complexity),
        np.argmax(result.complexity) / len(result.complexity),  # Normalized peak time
    ]

    return np.array(features)


def generate_complexity_dataset(
    n_qubits_list: List[int] = [4, 5, 6],
    n_hamiltonians: int = 50,
    t_max: float = 10.0,
    n_time_steps: int = 100
) -> List[Dict]:
    """
    Generate dataset of Krylov complexity curves.

    Args:
        n_qubits_list: List of system sizes to sample
        n_hamiltonians: Number of Hamiltonians per size
        t_max: Maximum evolution time
        n_time_steps: Number of time points

    Returns:
        List of dicts with Hamiltonian params and complexity results
    """
    dataset = []

    for n_qubits in n_qubits_list:
        for i in range(n_hamiltonians):
            # Random Hamiltonian parameters
            J_xy = np.random.uniform(0.5, 2.0)
            J_z = np.random.uniform(0.5, 2.0)
            h = np.random.uniform(0.0, 1.0)

            # Build Hamiltonian
            H = build_xxz_hamiltonian(n_qubits, J_xy, J_z, h)

            # Random initial state
            dim = 2 ** n_qubits
            psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi0 = psi0 / np.linalg.norm(psi0)

            # Compute complexity
            result = compute_krylov_complexity(H, psi0, t_max, n_time_steps)

            # Store data
            dataset.append({
                'n_qubits': n_qubits,
                'J_xy': J_xy,
                'J_z': J_z,
                'h': h,
                'times': result.times,
                'complexity': result.complexity,
                'features': compute_complexity_features(result),
                'krylov_dim': result.krylov_dimension,
                'growth_exp': result.growth_exponent,
                'saturation': result.saturation_value,
            })

    return dataset
