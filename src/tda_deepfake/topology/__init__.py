"""Persistent homology computation and vectorization.

Computes Vietoris-Rips persistent homology on point clouds (Ripser)
and converts persistence diagrams to fixed-size vectors for classification.
"""

from .persistent_homology import compute_persistence
from .vectorization import vectorize_diagrams

__all__ = ["compute_persistence", "vectorize_diagrams"]
