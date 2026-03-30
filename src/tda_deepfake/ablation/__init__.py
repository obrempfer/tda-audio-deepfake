"""Dimensional ablation module.

When a sample is flagged as a deepfake, dimensional ablation identifies
which feature groups drive the topological anomaly — converting a binary
detection into a traceable explanation grounded in speech production physics.
"""

from .ablation import AblationAnalyzer

__all__ = ["AblationAnalyzer"]
