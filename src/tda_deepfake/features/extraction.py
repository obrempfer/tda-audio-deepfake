"""Audio feature extraction for TDA deepfake detection.

Computes per-frame feature vectors from audio signals using a sliding-window
approach. Each feature dimension corresponds to a physically motivated property
of human speech production.

Minimal (class project) mode: 39-dim MFCC embedding (13 static + 13 Δ + 13 Δ²).
Extended mode: optional F0, jitter/shimmer/HNR, formants, spectral flux.
"""

import numpy as np
import numpy.typing as npt
from typing import Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import parselmouth
    from parselmouth.praat import call
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False

from ..config import AudioConfig, FeatureConfig


def extract_features(
    audio: npt.NDArray[np.float32],
    sample_rate: int = AudioConfig.SAMPLE_RATE,
    include_delta: bool = AudioConfig.INCLUDE_DELTA,
    include_delta2: bool = AudioConfig.INCLUDE_DELTA2,
    include_f0: bool = FeatureConfig.INCLUDE_F0,
    include_jitter_shimmer: bool = FeatureConfig.INCLUDE_JITTER_SHIMMER,
    include_formants: bool = FeatureConfig.INCLUDE_FORMANTS,
    include_spectral_flux: bool = FeatureConfig.INCLUDE_SPECTRAL_FLUX,
) -> npt.NDArray[np.float64]:
    """Compute the sliding-window feature matrix from a raw audio array.

    Each row is one frame (window); each column is one feature dimension.
    The resulting matrix is the point cloud for persistent homology.

    Args:
        audio: 1-D float32 array of audio samples.
        sample_rate: Sample rate of the audio signal.
        include_delta: Append first-derivative MFCCs.
        include_delta2: Append second-derivative MFCCs.
        include_f0: Append F0 and F0 slope columns (requires librosa).
        include_jitter_shimmer: Append jitter, shimmer, HNR (requires parselmouth).
        include_formants: Append F1–F3 frequencies and bandwidths (requires parselmouth).
        include_spectral_flux: Append frame-to-frame spectral change column.

    Returns:
        Feature matrix of shape (n_frames, n_dims). Rows are time-ordered frames.

    Raises:
        ImportError: If librosa is not installed (required for MFCCs).
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for feature extraction. pip install librosa")

    n_fft = int(sample_rate * AudioConfig.WINDOW_SIZE_MS / 1000)
    hop_length = int(sample_rate * AudioConfig.HOP_SIZE_MS / 1000)

    # --- Base: 13-dim MFCCs (static) ---
    mfccs = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=AudioConfig.N_MFCC,
        n_fft=n_fft,
        hop_length=hop_length,
    )  # shape: (n_mfcc, n_frames)

    features = [mfccs]

    # --- Delta MFCCs (rate of spectral change) ---
    if include_delta:
        delta = librosa.feature.delta(mfccs, order=1)
        features.append(delta)

    # --- Delta-delta MFCCs (acceleration / continuity) ---
    if include_delta2:
        delta2 = librosa.feature.delta(mfccs, order=2)
        features.append(delta2)

    # --- F0 and slope ---
    if include_f0:
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0).reshape(1, -1)
        f0_slope = np.gradient(f0, axis=1)
        features.extend([f0, f0_slope])

    # --- Spectral flux ---
    if include_spectral_flux:
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        flux = np.concatenate([[0.0], flux]).reshape(1, -1)
        features.append(flux)

    # Stack all librosa features: shape (total_dims, n_frames)
    feature_matrix = np.vstack(features).T  # shape: (n_frames, total_dims)

    # --- Praat-based features (per-frame interpolation from global measurements) ---
    if include_jitter_shimmer or include_formants:
        feature_matrix = _append_praat_features(
            feature_matrix,
            audio,
            sample_rate,
            hop_length,
            include_jitter_shimmer=include_jitter_shimmer,
            include_formants=include_formants,
        )

    return feature_matrix.astype(np.float64)


def build_point_cloud(feature_matrix: npt.NDArray) -> npt.NDArray:
    """Return the feature matrix as a point cloud for PH computation.

    Currently a pass-through; may add optional standardization or
    dimensionality reduction in future.

    Args:
        feature_matrix: Output of extract_features(), shape (n_frames, n_dims).

    Returns:
        Point cloud array of shape (n_points, n_dims).
    """
    return feature_matrix


def _append_praat_features(
    feature_matrix: npt.NDArray,
    audio: npt.NDArray,
    sample_rate: int,
    hop_length: int,
    include_jitter_shimmer: bool,
    include_formants: bool,
) -> npt.NDArray:
    """Append Praat-derived voice quality features to the feature matrix.

    Args:
        feature_matrix: Existing feature matrix (n_frames, existing_dims).
        audio: Raw audio samples.
        sample_rate: Audio sample rate.
        hop_length: Hop size in samples (used to compute frame timestamps).
        include_jitter_shimmer: Append jitter, shimmer, HNR columns.
        include_formants: Append F1–F3 frequency and bandwidth columns.

    Returns:
        Feature matrix with additional Praat columns appended.
    """
    if not PARSELMOUTH_AVAILABLE:
        raise ImportError(
            "praat-parselmouth is required for jitter/shimmer/formant features. "
            "pip install praat-parselmouth"
        )

    n_frames = feature_matrix.shape[0]
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    extra_cols = []

    if include_jitter_shimmer:
        # PointProcess for jitter/shimmer
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr_obj = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(hnr_obj, "Get mean", 0, 0)

        # Broadcast scalar measurements across all frames
        extra_cols.append(np.full((n_frames, 1), jitter if jitter == jitter else 0.0))
        extra_cols.append(np.full((n_frames, 1), shimmer if shimmer == shimmer else 0.0))
        extra_cols.append(np.full((n_frames, 1), hnr if hnr == hnr else 0.0))

    if include_formants:
        formant_obj = call(snd, "To Formant (burg)", 0, FeatureConfig.N_FORMANTS, 5500, 0.025, 50)
        times = np.arange(n_frames) * hop_length / sample_rate
        f_cols = np.zeros((n_frames, FeatureConfig.N_FORMANTS * 2))  # freq + bandwidth per formant
        for i, t in enumerate(times):
            for fn in range(1, FeatureConfig.N_FORMANTS + 1):
                freq = call(formant_obj, "Get value at time", fn, t, "Hertz", "Linear")
                bw = call(formant_obj, "Get bandwidth at time", fn, t, "Hertz", "Linear")
                f_cols[i, (fn - 1) * 2] = freq if freq == freq else 0.0
                f_cols[i, (fn - 1) * 2 + 1] = bw if bw == bw else 0.0
        extra_cols.append(f_cols)

    return np.hstack([feature_matrix] + extra_cols)
