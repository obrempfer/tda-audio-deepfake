"""Persistent homology computation and vectorization.

Supports Vietoris-Rips persistent homology on point clouds and cubical
persistent homology on grid representations, then vectorizes diagrams
for downstream classification.
"""

from .persistent_homology import compute_persistence
from .vectorization import vectorize_diagrams

__all__ = ["compute_persistence", "vectorize_diagrams"]
