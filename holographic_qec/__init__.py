"""
Holographic Quantum Error Correction Package

A unified framework for learning holographic quantum codes combining:
- GNN-based code design (P1)
- DeepONet-based Krylov complexity learning (P3)
- Geometry-complexity correlation analysis

Author: PhD Research Project
"""

__version__ = "0.1.0"

from . import codes
from . import gnn
from . import dynamics
from . import analysis
from . import utils
