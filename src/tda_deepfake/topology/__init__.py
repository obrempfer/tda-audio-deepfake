"""Topological and topology-inspired feature extraction.

Supports Vietoris-Rips persistent homology on point clouds and cubical
persistent homology on grid representations, then vectorizes diagrams
for downstream classification. Also includes a discrete Morse-Smale-inspired
signature on scalar fields.
"""

from .persistent_homology import compute_persistence
from .morse_smale import compute_morse_smale_signature
from .takens import build_takens_embedding, build_takens_signal
from .vectorization import flatten_vector_blocks, vectorize_diagram_blocks, vectorize_diagrams

__all__ = [
    "compute_persistence",
    "compute_morse_smale_signature",
    "build_takens_signal",
    "build_takens_embedding",
    "vectorize_diagrams",
    "vectorize_diagram_blocks",
    "flatten_vector_blocks",
]
