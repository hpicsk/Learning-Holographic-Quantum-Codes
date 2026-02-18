"""
Quantum Error Correcting Code Generation Module

Provides:
- HaPPY code construction
- Stabilizer code utilities
- Dataset generation for training
"""

from .happy_codes import (
    create_happy_code,
    create_perfect_tensor,
    HaPPYCode,
)
from .stabilizer import (
    StabilizerCode,
    pauli_weight,
    compute_code_distance,
)
