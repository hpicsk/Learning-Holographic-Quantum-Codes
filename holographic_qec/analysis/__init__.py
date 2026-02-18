"""
Analysis Module for Geometry-Dynamics Correlation

Provides:
- Statistical correlation analysis
- AdS/CFT dictionary tests
- Phase transition detection
"""

from .correlation import GeometryComplexityAnalyzer, PhaseTransitionAnalyzer
from .holographic import HolographicDictionary, HolographicTestResult

__all__ = [
    'GeometryComplexityAnalyzer',
    'PhaseTransitionAnalyzer',
    'HolographicDictionary',
    'HolographicTestResult'
]
