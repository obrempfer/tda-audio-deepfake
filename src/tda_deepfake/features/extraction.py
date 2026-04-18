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
from scipy.ndimage import gaussian_filter

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

from ..config import AudioConfig, FeatureConfig, PointCloudConfig, SpectrogramConfig


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


def build_mel_spectrogram(
    audio: npt.NDArray[np.float32],
    sample_rate: int = AudioConfig.SAMPLE_RATE,
    n_mels: int = SpectrogramConfig.N_MELS,
    power: float = SpectrogramConfig.POWER,
    fmin: float = SpectrogramConfig.FMIN,
    fmax: Optional[float] = SpectrogramConfig.FMAX,
    log_scale: bool = SpectrogramConfig.LOG_SCALE,
    compression: str = SpectrogramConfig.COMPRESSION,
    smoothing: str = SpectrogramConfig.SMOOTHING,
    smoothing_sigma: float = SpectrogramConfig.SMOOTHING_SIGMA,
    smoothing_axis: str = SpectrogramConfig.SMOOTHING_AXIS,
    band_mask_mode: str = SpectrogramConfig.BAND_MASK_MODE,
    band_split_low: float = SpectrogramConfig.BAND_SPLIT_LOW,
    band_split_high: float = SpectrogramConfig.BAND_SPLIT_HIGH,
    band_mask_fill: str = SpectrogramConfig.BAND_MASK_FILL,
    temporal_field_mode: str = SpectrogramConfig.TEMPORAL_FIELD_MODE,
    temporal_field_sigma: float = SpectrogramConfig.TEMPORAL_FIELD_SIGMA,
    energy_weighting_mode: str = SpectrogramConfig.ENERGY_WEIGHTING_MODE,
    energy_weighting_gamma: float = SpectrogramConfig.ENERGY_WEIGHTING_GAMMA,
    energy_gate_percentile: Optional[float] = SpectrogramConfig.ENERGY_GATE_PERCENTILE,
    energy_gate_fill: str = SpectrogramConfig.ENERGY_GATE_FILL,
    normalize: bool = SpectrogramConfig.NORMALIZE,
    normalization_method: str = SpectrogramConfig.NORMALIZATION_METHOD,
    max_frames: Optional[int] = SpectrogramConfig.MAX_FRAMES,
) -> npt.NDArray[np.float64]:
    """Build a mel-spectrogram grid for cubical persistent homology."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for mel spectrogram extraction. pip install librosa")

    n_fft = int(sample_rate * AudioConfig.WINDOW_SIZE_MS / 1000)
    hop_length = int(sample_rate * AudioConfig.HOP_SIZE_MS / 1000)

    grid = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=power,
        fmin=fmin,
        fmax=fmax,
    ).astype(np.float64)

    grid = _apply_band_mask(
        grid,
        mode=band_mask_mode,
        split_low=band_split_low,
        split_high=band_split_high,
        fill=band_mask_fill,
    )

    grid = _apply_energy_gate(
        grid,
        percentile=energy_gate_percentile,
        fill=energy_gate_fill,
    )

    grid = _compress_grid(
        grid,
        compression=compression,
        log_scale=log_scale,
    )

    grid = _apply_temporal_field_transform(
        grid,
        mode=temporal_field_mode,
        sigma=temporal_field_sigma,
    )

    grid = _apply_energy_weighting(
        grid,
        mode=energy_weighting_mode,
        gamma=energy_weighting_gamma,
    )

    grid = _smooth_grid(
        grid,
        method=smoothing,
        sigma=smoothing_sigma,
        axis=smoothing_axis,
    )

    if max_frames is not None and grid.shape[1] > max_frames:
        indices = np.linspace(0, grid.shape[1] - 1, max_frames, dtype=int)
        grid = grid[:, indices]

    if normalize:
        grid = _normalize_grid(grid, method=normalization_method)

    return grid


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


def _normalize_grid(grid: npt.NDArray, method: str = "minmax") -> npt.NDArray:
    """Normalize a 2-D spectrogram grid before cubical persistent homology."""
    if method == "none":
        return grid
    if method == "zscore":
        mean = np.mean(grid)
        std = np.std(grid)
        if std == 0.0:
            std = 1.0
        return (grid - mean) / std
    if method == "minmax":
        lower = np.min(grid)
        upper = np.max(grid)
        if upper == lower:
            return np.zeros_like(grid)
        return (grid - lower) / (upper - lower)
    raise ValueError(f"Unknown grid normalization method: {method!r}")


def _smooth_grid(
    grid: npt.NDArray,
    method: str = "none",
    sigma: float = 1.0,
    axis: str = "both",
) -> npt.NDArray:
    """Apply optional smoothing to a spectrogram grid before cubical PH."""
    if method == "none":
        return grid
    if sigma <= 0:
        raise ValueError(f"smoothing sigma must be positive, got {sigma}")
    if method == "gaussian":
        axis_mode = axis.lower()
        if axis_mode == "both":
            sigma_vec = (float(sigma), float(sigma))
        elif axis_mode == "frequency":
            sigma_vec = (float(sigma), 0.0)
        elif axis_mode == "time":
            sigma_vec = (0.0, float(sigma))
        else:
            raise ValueError(f"Unknown smoothing axis mode: {axis!r}")
        return gaussian_filter(grid, sigma=sigma_vec, mode="nearest")
    raise ValueError(f"Unknown spectrogram smoothing method: {method!r}")


def _compress_grid(
    grid: npt.NDArray,
    compression: str = "auto",
    log_scale: bool = True,
) -> npt.NDArray:
    """Apply dynamic-range compression to a spectrogram grid."""
    mode = (compression or "auto").lower()
    if mode == "auto":
        mode = "db" if log_scale else "none"

    if mode == "none":
        return grid
    if mode == "db":
        return librosa.power_to_db(grid, ref=np.max)
    if mode == "log1p":
        return np.log1p(np.maximum(grid, 0.0))
    if mode == "root":
        return np.sqrt(np.maximum(grid, 0.0))
    raise ValueError(f"Unknown spectrogram compression mode: {compression!r}")


def _apply_energy_gate(
    grid: npt.NDArray,
    percentile: Optional[float] = None,
    fill: str = "zero",
) -> npt.NDArray:
    """Suppress low-energy frames as a simple voiced-region gate."""
    if percentile is None:
        return grid
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(f"energy gate percentile must be in [0, 100], got {percentile}")

    frame_energy = np.mean(grid, axis=0)
    threshold = np.percentile(frame_energy, percentile)
    mask = frame_energy < threshold
    if not np.any(mask):
        return grid

    out = grid.copy()
    fill_mode = fill.lower()
    if fill_mode == "zero":
        fill_value = 0.0
    elif fill_mode == "min":
        fill_value = float(np.min(grid))
    else:
        raise ValueError(f"Unknown energy gate fill mode: {fill!r}")

    out[:, mask] = fill_value
    return out


def _apply_band_mask(
    grid: npt.NDArray,
    mode: str = "none",
    split_low: float = 0.33,
    split_high: float = 0.66,
    fill: str = "zero",
) -> npt.NDArray:
    """Apply optional mel-band masking for frequency-region ablations."""
    ablation_mode = (mode or "none").lower()
    if ablation_mode == "none":
        return grid
    valid_modes = {
        "drop_low",
        "drop_mid",
        "drop_high",
        "keep_low",
        "keep_mid",
        "keep_high",
    }
    if ablation_mode not in valid_modes:
        raise ValueError(f"Unknown band mask mode: {mode!r}")
    if not (0.0 < split_low < split_high < 1.0):
        raise ValueError(
            "band split fractions must satisfy 0 < split_low < split_high < 1; "
            f"got split_low={split_low}, split_high={split_high}"
        )

    n_mels = grid.shape[0]
    low_end = int(np.floor(split_low * n_mels))
    high_start = int(np.floor(split_high * n_mels))
    # Keep all segments non-empty for predictable ablations.
    low_end = max(1, min(low_end, n_mels - 2))
    high_start = max(low_end + 1, min(high_start, n_mels - 1))

    low = slice(0, low_end)
    mid = slice(low_end, high_start)
    high = slice(high_start, n_mels)

    fill_mode = fill.lower()
    if fill_mode == "zero":
        fill_value = 0.0
    elif fill_mode == "min":
        fill_value = float(np.min(grid))
    else:
        raise ValueError(f"Unknown band mask fill mode: {fill!r}")

    out = grid.copy()

    if ablation_mode == "drop_low":
        out[low, :] = fill_value
    elif ablation_mode == "drop_mid":
        out[mid, :] = fill_value
    elif ablation_mode == "drop_high":
        out[high, :] = fill_value
    elif ablation_mode == "keep_low":
        out[mid, :] = fill_value
        out[high, :] = fill_value
    elif ablation_mode == "keep_mid":
        out[low, :] = fill_value
        out[high, :] = fill_value
    elif ablation_mode == "keep_high":
        out[low, :] = fill_value
        out[mid, :] = fill_value

    return out


def _apply_temporal_field_transform(
    grid: npt.NDArray,
    mode: str = "none",
    sigma: float = 2.0,
) -> npt.NDArray:
    """Apply optional temporal-structure transform on the spectrogram field."""
    transform_mode = (mode or "none").lower()
    if transform_mode == "none":
        return grid
    if sigma <= 0:
        raise ValueError(f"temporal field sigma must be positive, got {sigma}")

    smoothed_time = gaussian_filter(grid, sigma=(0.0, float(sigma)), mode="nearest")
    if transform_mode == "sustained":
        return smoothed_time
    if transform_mode == "transition":
        return np.abs(grid - smoothed_time)
    raise ValueError(f"Unknown temporal field mode: {mode!r}")


def _apply_energy_weighting(
    grid: npt.NDArray,
    mode: str = "none",
    gamma: float = 1.0,
) -> npt.NDArray:
    """Apply optional frame-energy weighting on a spectrogram field."""
    weighting_mode = (mode or "none").lower()
    if weighting_mode == "none":
        return grid
    if weighting_mode != "power":
        raise ValueError(f"Unknown energy weighting mode: {mode!r}")
    if gamma <= 0:
        raise ValueError(f"energy weighting gamma must be positive, got {gamma}")

    shifted = grid - np.min(grid)
    frame_energy = np.mean(shifted, axis=0)
    scale = float(np.max(frame_energy))
    if scale <= 0.0:
        return grid
    normalized = frame_energy / scale
    weights = normalized ** float(gamma)
    return grid * weights[np.newaxis, :]


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
