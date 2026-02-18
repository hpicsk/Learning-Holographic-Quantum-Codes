"""
Quantum Dynamics and Complexity Module

Provides:
- Krylov complexity computation (Lanczos algorithm)
- DeepONet for learning dynamics
- Hamiltonian construction utilities
"""

from .krylov import (
    compute_krylov_basis,
    compute_krylov_complexity,
    extract_scaling_exponent,
)
