"""Audio feature extraction module.

Computes physically motivated embeddings from audio signals:
- MFCCs and their time derivatives (librosa)
- F0 and pitch dynamics
- Jitter, shimmer, HNR (praat-parselmouth)
- Formant frequencies and bandwidths (praat-parselmouth)
- Spectral flux
"""

from .extraction import extract_features, build_point_cloud

__all__ = ["extract_features", "build_point_cloud"]
