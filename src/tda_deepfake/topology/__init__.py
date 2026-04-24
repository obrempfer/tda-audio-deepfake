"""Topological and topology-inspired feature extraction.

Supports Vietoris-Rips persistent homology on point clouds and cubical
persistent homology on grid representations, then vectorizes diagrams
for downstream classification. Also includes a discrete Morse-Smale-inspired
signature on scalar fields.
"""

from .persistent_homology import compute_persistence
from .morse_smale import compute_morse_smale_signature
from .vectorization import flatten_vector_blocks, vectorize_diagram_blocks, vectorize_diagrams

__all__ = [
    "compute_persistence",
    "compute_morse_smale_signature",
    "vectorize_diagrams",
    "vectorize_diagram_blocks",
    "flatten_vector_blocks",
]
