"""Configuration for the TDA audio deepfake detection pipeline.

Each config class corresponds to a pipeline stage. Modify class
attributes directly or call the helper functions at runtime.
"""

from typing import Optional

import yaml


class AudioConfig:
    """Audio loading and framing parameters.

    Attributes:
        SAMPLE_RATE: Target sample rate for all audio (Hz).
        WINDOW_SIZE_MS: STFT window length in milliseconds.
        HOP_SIZE_MS: STFT hop length in milliseconds.
        N_MFCC: Number of MFCC coefficients (static layer).
        INCLUDE_DELTA: Whether to append first-derivative MFCCs.
        INCLUDE_DELTA2: Whether to append second-derivative MFCCs.
    """
    SAMPLE_RATE: int = 16000
    WINDOW_SIZE_MS: float = 25.0
    HOP_SIZE_MS: float = 10.0
    N_MFCC: int = 13
    INCLUDE_DELTA: bool = True
    INCLUDE_DELTA2: bool = True


class FeatureConfig:
    """Optional additional feature dimensions.

    These are off by default for the minimal class project (39-dim MFCC only).
    Enable selectively to add physically motivated dimensions.

    Attributes:
        INCLUDE_F0: Include fundamental frequency (F0) and slope.
        INCLUDE_JITTER_SHIMMER: Include jitter, shimmer, HNR via Praat.
        INCLUDE_FORMANTS: Include F1–F3 formant frequencies and bandwidths.
        INCLUDE_SPECTRAL_FLUX: Include frame-to-frame spectral change.
        N_FORMANTS: Number of formants to extract (max 3 recommended).
    """
    INCLUDE_F0: bool = False
    INCLUDE_JITTER_SHIMMER: bool = False
    INCLUDE_FORMANTS: bool = False
    INCLUDE_SPECTRAL_FLUX: bool = False
    N_FORMANTS: int = 3


class PointCloudConfig:
    """Point-cloud preprocessing before persistent homology.

    Attributes:
        NORMALIZE: Whether to normalize feature dimensions before PH.
        NORMALIZATION_METHOD: Normalization strategy ('zscore' or 'none').
        PROJECTION: Optional dimensionality reduction before PH ('none', 'pca', 'jl').
        PROJECTION_DIM: Target dimensionality for PCA/JL projection.
        PROJECTION_RANDOM_STATE: Random seed for stochastic projections.
    """
    NORMALIZE: bool = False
    NORMALIZATION_METHOD: str = "zscore"
    PROJECTION: str = "none"
    PROJECTION_DIM: Optional[int] = None
    PROJECTION_RANDOM_STATE: int = 42


class SpectrogramConfig:
    """Grid representation parameters for cubical persistent homology.

    Attributes:
        KIND: Spectrogram family to build ('mel' for now).
        N_MELS: Number of mel bins in the time-frequency grid.
        POWER: Exponent for mel-spectrogram magnitude construction.
        FMIN: Lower frequency bound for the mel filterbank (Hz).
        FMAX: Upper frequency bound for the mel filterbank (Hz or None).
        LOG_SCALE: Legacy toggle for dB scaling (kept for backward compatibility).
        COMPRESSION: Dynamic-range compression ('auto', 'none', 'db', 'log1p', 'root').
        SMOOTHING: Optional spectrogram smoothing ('none' or 'gaussian').
        SMOOTHING_SIGMA: Standard deviation for Gaussian smoothing.
        SMOOTHING_AXIS: Where smoothing is applied ('both', 'time', 'frequency').
        BAND_MASK_MODE: Optional mel-band masking mode
            ('none', 'drop_low', 'drop_mid', 'drop_high', 'keep_low', 'keep_mid', 'keep_high').
        BAND_SPLIT_LOW: Fractional mel-bin boundary for low/mid split.
        BAND_SPLIT_HIGH: Fractional mel-bin boundary for mid/high split.
        BAND_MASK_FILL: Fill strategy for masked mel bands ('zero' or 'min').
        TEMPORAL_FIELD_MODE: Optional time-structure transform
            ('none', 'transition', 'sustained').
        TEMPORAL_FIELD_SIGMA: Time-axis smoothing sigma for temporal transforms.
        ENERGY_WEIGHTING_MODE: Optional frame-energy weighting mode ('none', 'power').
        ENERGY_WEIGHTING_GAMMA: Exponent for power energy weighting.
        ENERGY_GATE_PERCENTILE: Optional frame-energy percentile gate (0-100, None disables).
        ENERGY_GATE_FILL: Fill strategy for gated frames ('zero' or 'min').
        NORMALIZE: Whether to normalize the grid before cubical PH.
        NORMALIZATION_METHOD: Grid normalization strategy ('minmax', 'zscore', 'none').
        MAX_FRAMES: Optional cap on time frames via uniform subsampling.
    """
    KIND: str = "mel"
    N_MELS: int = 64
    POWER: float = 2.0
    FMIN: float = 0.0
    FMAX: Optional[float] = None
    LOG_SCALE: bool = True
    COMPRESSION: str = "auto"
    SMOOTHING: str = "none"
    SMOOTHING_SIGMA: float = 1.0
    SMOOTHING_AXIS: str = "both"
    BAND_MASK_MODE: str = "none"
    BAND_SPLIT_LOW: float = 0.33
    BAND_SPLIT_HIGH: float = 0.66
    BAND_MASK_FILL: str = "zero"
    TEMPORAL_FIELD_MODE: str = "none"
    TEMPORAL_FIELD_SIGMA: float = 2.0
    ENERGY_WEIGHTING_MODE: str = "none"
    ENERGY_WEIGHTING_GAMMA: float = 1.0
    ENERGY_GATE_PERCENTILE: Optional[float] = None
    ENERGY_GATE_FILL: str = "zero"
    NORMALIZE: bool = True
    NORMALIZATION_METHOD: str = "minmax"
    MAX_FRAMES: Optional[int] = 256


class MorseSmaleConfig:
    """Discrete Morse-Smale-inspired feature extraction on spectrogram grids.

    Preferred path uses topopy's approximate Morse-Smale complex. A local
    discrete fallback is available when topopy is unavailable.

    Attributes:
        IMPLEMENTATION: 'topopy' or 'approx'.
        GRAPH_MAX_NEIGHBORS: Neighborhood size for topopy's graph construction.
        GRAPH_RELAXED: Whether topopy/nglpy uses the relaxed empty-region graph.
        NORMALIZATION: Optional topopy normalization mode ('feature', 'zscore', or None).
        SIMPLIFICATION: topopy simplification mode.
        NEIGHBORHOOD_SIZE: Window size for local extrema detection.
        TOP_K_BASINS: Number of largest ascending/descending basins to keep.
        INCLUDE_EXTREMA_VALUES: Whether to include strongest extrema values.
        TOPO_K_EXTREMA: Number of strongest minima/maxima values to keep.
    """
    IMPLEMENTATION: str = "topopy"
    GRAPH_MAX_NEIGHBORS: int = 8
    GRAPH_RELAXED: bool = False
    NORMALIZATION: Optional[str] = None
    SIMPLIFICATION: str = "difference"
    NEIGHBORHOOD_SIZE: int = 3
    TOP_K_BASINS: int = 8
    INCLUDE_EXTREMA_VALUES: bool = True
    TOP_K_EXTREMA: int = 8


class TakensConfig:
    """Time-delay embedding parameters for scalar audio signals.

    Attributes:
        SIGNAL_TYPE: Scalar signal family ('low_wave', 'low_env', 'full_wave', 'full_env').
        LOWPASS_CUTOFF_HZ: Optional low-pass cutoff for low-band waveform extraction.
            When None, a cutoff is derived from the low-band split fraction.
        FILTER_ORDER: Butterworth low-pass order for low-band waveform extraction.
        SIGNAL_NORMALIZATION: Per-sample signal normalization ('none' or 'zscore').
        ENVELOPE_COMPRESSION: Optional envelope compression ('none' or 'log1p').
        ENVELOPE_SMOOTH_SIGMA: Gaussian smoothing sigma in envelope frames.
        EMBEDDING_DIM: Takens embedding dimension.
        DELAY: Delay in samples of the stride-subsampled signal.
        STRIDE: Uniform decimation applied before delay embedding.
        MAX_POINTS: Optional cap on Takens points before PH.
    """
    SIGNAL_TYPE: str = "low_wave"
    LOWPASS_CUTOFF_HZ: Optional[float] = None
    FILTER_ORDER: int = 5
    SIGNAL_NORMALIZATION: str = "zscore"
    ENVELOPE_COMPRESSION: str = "log1p"
    ENVELOPE_SMOOTH_SIGMA: float = 1.0
    EMBEDDING_DIM: int = 5
    DELAY: int = 4
    STRIDE: int = 80
    MAX_POINTS: Optional[int] = 300


class TopologyConfig:
    """Persistent homology computation parameters.

    Attributes:
        COMPLEX: Topological complex family ('vietoris_rips', 'cubical', 'knn_flag',
            'morse_smale', 'morse_smale_approx', or 'takens_ph').
        MAX_HOMOLOGY_DIM: Highest homological dimension to compute (0=H0, 1=H1).
        DISTANCE_METRIC: Distance metric for Vietoris-Rips ('euclidean' or 'precomputed').
        CUBICAL_FILTRATION: Cubical filtration polarity ('sublevel' or 'superlevel').
        KNN_K: Number of neighbors for the kNN graph used by the flag/clique complex.
        KNN_GRAPH_MODE: Symmetrization mode for the kNN graph ('union' or 'mutual').
        MAX_EDGE_LENGTH: Maximum filtration value (None = auto).
        COEFF: Coefficient field for homology computation.
    """
    COMPLEX: str = "vietoris_rips"
    MAX_HOMOLOGY_DIM: int = 1
    DISTANCE_METRIC: str = "euclidean"
    CUBICAL_FILTRATION: str = "superlevel"
    KNN_K: int = 15
    KNN_GRAPH_MODE: str = "union"
    MAX_EDGE_LENGTH: Optional[float] = None
    COEFF: int = 2


class VectorizationConfig:
    """Persistence diagram vectorization parameters.

    Attributes:
        METHOD: Vectorization method ('persistence_image' or 'landscape').
        PI_N_BINS: Grid resolution for persistence images (n_bins x n_bins).
        PI_SIGMA: Gaussian kernel bandwidth for persistence images.
        PI_WEIGHT: Weight function ('linear' or 'persistence').
        LANDSCAPE_N_LAYERS: Number of landscape layers to compute.
        LANDSCAPE_N_BINS: Resolution of each landscape layer.
        HOMOLOGY_WEIGHTS: Optional per-dimension scaling applied after
            vectorization (e.g., [1.0, 0.5] scales H1 by 0.5).
    """
    METHOD: str = "persistence_image"
    PI_N_BINS: int = 20
    PI_SIGMA: float = 0.1
    PI_WEIGHT: str = "linear"
    LANDSCAPE_N_LAYERS: int = 5
    LANDSCAPE_N_BINS: int = 100
    HOMOLOGY_WEIGHTS: Optional[list[float]] = None


class ClassifierConfig:
    """Classifier training parameters.

    Attributes:
        MODEL: Classifier type ('svm' or 'logistic').
        SVM_KERNEL: SVM kernel type.
        SVM_C: SVM regularization parameter.
        SCALE_FEATURES: Whether to apply StandardScaler before the classifier.
        CV_FOLDS: Number of cross-validation folds.
        RANDOM_STATE: Random seed for reproducibility.
    """
    MODEL: str = "svm"
    SVM_KERNEL: str = "rbf"
    SVM_C: float = 1.0
    SCALE_FEATURES: bool = True
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42


class AblationConfig:
    """Dimensional ablation configuration.

    Feature groups are defined as named slices of the embedding vector.
    The groups should be updated to match the actual embedding dimensionality.

    Attributes:
        FEATURE_GROUPS: Dict mapping group name → list of dimension indices.
        ANOMALY_SCORE_THRESHOLD: Minimum anomaly score to trigger ablation.
    """
    FEATURE_GROUPS: dict = {
        "mfcc_static": list(range(0, 13)),
        "mfcc_delta": list(range(13, 26)),
        "mfcc_delta2": list(range(26, 39)),
        # Extended groups (populated at runtime when optional features are enabled)
        # "f0": [...],
        # "jitter_shimmer_hnr": [...],
        # "formants": [...],
        # "spectral_flux": [...],
    }
    ANOMALY_SCORE_THRESHOLD: float = 0.5


# Runtime configuration helpers
_KEY_MAP = {
    "audio": AudioConfig,
    "feature": FeatureConfig,
    "point_cloud": PointCloudConfig,
    "spectrogram": SpectrogramConfig,
    "morse_smale": MorseSmaleConfig,
    "takens": TakensConfig,
    "topology": TopologyConfig,
    "vectorization": VectorizationConfig,
    "classifier": ClassifierConfig,
    "ablation": AblationConfig,
}


def export_runtime_config() -> dict[str, dict[str, object]]:
    """Return a serializable snapshot of all runtime config classes."""
    snapshot: dict[str, dict[str, object]] = {}
    for section, cls in _KEY_MAP.items():
        values = {}
        for name, value in vars(cls).items():
            if name.startswith("_") or callable(value):
                continue
            values[name.lower()] = value
        snapshot[section] = values
    return snapshot


def apply_runtime_config(snapshot: dict[str, dict[str, object]]) -> None:
    """Apply a config snapshot produced by export_runtime_config()."""
    for section, values in snapshot.items():
        cls = _KEY_MAP.get(section)
        if cls is None:
            continue
        for key, value in values.items():
            attr = key.upper()
            if hasattr(cls, attr):
                setattr(cls, attr, value)

def configure_audio(sample_rate: Optional[int] = None, n_mfcc: Optional[int] = None) -> None:
    """Update AudioConfig at runtime.

    Args:
        sample_rate: Override target sample rate.
        n_mfcc: Override number of MFCC coefficients.
    """
    if sample_rate is not None:
        AudioConfig.SAMPLE_RATE = sample_rate
    if n_mfcc is not None:
        AudioConfig.N_MFCC = n_mfcc


def load_config_from_yaml(yaml_path: str) -> None:
    """Load configuration from a YAML file and update config class attributes.

    Supported top-level keys: audio, feature, point_cloud, spectrogram, morse_smale,
    takens, topology, vectorization, classifier, ablation.
    Each key maps to a dict of attribute names (lowercase) and their values.

    Args:
        yaml_path: Path to YAML configuration file.

    Example YAML::

        audio:
          sample_rate: 16000
          n_mfcc: 13
        topology:
          max_homology_dim: 1
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        return

    for section, cls in _KEY_MAP.items():
        if section not in cfg:
            continue
        for key, value in cfg[section].items():
            attr = key.upper()
            if hasattr(cls, attr):
                setattr(cls, attr, value)
