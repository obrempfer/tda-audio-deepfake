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

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

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

from ..config import AudioConfig, FeatureConfig, PointCloudConfig


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


def build_point_cloud(
    feature_matrix: npt.NDArray,
    max_points: Optional[int] = 300,
    normalize: bool = PointCloudConfig.NORMALIZE,
    normalization_method: str = PointCloudConfig.NORMALIZATION_METHOD,
    projection: str = PointCloudConfig.PROJECTION,
    projection_dim: Optional[int] = PointCloudConfig.PROJECTION_DIM,
    projection_random_state: int = PointCloudConfig.PROJECTION_RANDOM_STATE,
) -> npt.NDArray:
    """Return the feature matrix as a point cloud for PH computation.

    Subsamples uniformly to at most max_points rows to keep Ripser tractable
    on long utterances (PH runtime scales super-linearly with n_points).

    Args:
        feature_matrix: Output of extract_features(), shape (n_frames, n_dims).
        max_points: Maximum number of points to retain. None disables subsampling.
        normalize: Whether to normalize feature dimensions before PH.
        normalization_method: Dimensional normalization strategy.
        projection: Optional projection method ('none', 'pca', 'jl').
        projection_dim: Target dimensionality for projection.
        projection_random_state: Random seed for stochastic projection methods.

    Returns:
        Point cloud array of shape (n_points, n_dims).
    """
    point_cloud = feature_matrix
    if max_points is not None and len(feature_matrix) > max_points:
        indices = np.linspace(0, len(feature_matrix) - 1, max_points, dtype=int)
        point_cloud = feature_matrix[indices]

    if normalize:
        point_cloud = _normalize_point_cloud(point_cloud, method=normalization_method)

    if projection != "none":
        point_cloud = _project_point_cloud(
            point_cloud,
            method=projection,
            target_dim=projection_dim,
            random_state=projection_random_state,
        )

    return point_cloud


def _normalize_point_cloud(point_cloud: npt.NDArray, method: str = "zscore") -> npt.NDArray:
    """Normalize point-cloud feature dimensions before persistent homology."""
    if method == "none":
        return point_cloud
    if method != "zscore":
        raise ValueError(f"Unknown normalization method: {method!r}")

    mean = np.mean(point_cloud, axis=0, keepdims=True)
    std = np.std(point_cloud, axis=0, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (point_cloud - mean) / std


def _project_point_cloud(
    point_cloud: npt.NDArray,
    method: str,
    target_dim: Optional[int],
    random_state: int,
) -> npt.NDArray:
    """Apply optional dimensionality reduction before persistent homology."""
    if target_dim is None:
        return point_cloud
    if target_dim <= 0:
        raise ValueError(f"projection_dim must be positive, got {target_dim}")
    if target_dim >= point_cloud.shape[1]:
        return point_cloud

    if method == "pca":
        n_components = min(target_dim, point_cloud.shape[0], point_cloud.shape[1])
        if n_components >= point_cloud.shape[1]:
            return point_cloud
        projector = PCA(n_components=n_components, random_state=random_state)
        return projector.fit_transform(point_cloud)

    if method == "jl":
        projector = GaussianRandomProjection(
            n_components=target_dim,
            random_state=random_state,
        )
        return projector.fit_transform(point_cloud)

    raise ValueError(f"Unknown projection method: {method!r}")


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
