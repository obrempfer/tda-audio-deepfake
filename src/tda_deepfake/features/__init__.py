"""Audio feature extraction module.

Computes physically motivated embeddings from audio signals:
- MFCCs and their time derivatives (librosa)
- F0 and pitch dynamics
- Jitter, shimmer, HNR (praat-parselmouth)
- Formant frequencies and bandwidths (praat-parselmouth)
- Spectral flux
"""

from .extraction import (
    build_mel_spectrogram,
    build_point_cloud,
    build_raw_mel_spectrogram,
    extract_features,
    postprocess_mel_spectrogram,
)

__all__ = [
    "extract_features",
    "build_point_cloud",
    "build_mel_spectrogram",
    "build_raw_mel_spectrogram",
    "postprocess_mel_spectrogram",
]
