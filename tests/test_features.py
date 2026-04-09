"""Tests for audio feature extraction."""

import numpy as np

from tda_deepfake.features.extraction import extract_features, build_point_cloud
from tda_deepfake.config import AudioConfig


def _synthetic_audio(duration_s: float = 1.0, sr: int = 16000) -> np.ndarray:
    """Generate a simple sine wave as synthetic audio."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_extract_features_shape_minimal():
    audio = _synthetic_audio()
    features = extract_features(audio, include_delta=True, include_delta2=True,
                                 include_f0=False, include_jitter_shimmer=False,
                                 include_formants=False, include_spectral_flux=False)
    # Should be 39-dimensional (13 static + 13 delta + 13 delta2)
    assert features.ndim == 2
    assert features.shape[1] == AudioConfig.N_MFCC * 3


def test_extract_features_shape_static_only():
    audio = _synthetic_audio()
    features = extract_features(audio, include_delta=False, include_delta2=False,
                                 include_f0=False, include_jitter_shimmer=False,
                                 include_formants=False, include_spectral_flux=False)
    assert features.shape[1] == AudioConfig.N_MFCC


def test_build_point_cloud_passthrough():
    audio = _synthetic_audio()
    features = extract_features(audio)
    cloud = build_point_cloud(features)
    np.testing.assert_array_equal(cloud, features)


def test_no_nan_in_features():
    audio = _synthetic_audio()
    features = extract_features(audio)
    assert not np.any(np.isnan(features))


def test_build_point_cloud_normalization_zscore():
    rng = np.random.default_rng(0)
    features = rng.normal(loc=10.0, scale=5.0, size=(100, 4))
    cloud = build_point_cloud(features, max_points=None, normalize=True)

    np.testing.assert_allclose(np.mean(cloud, axis=0), np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(np.std(cloud, axis=0), np.ones(4), atol=1e-7)


def test_build_point_cloud_pca_projection_reduces_dimension():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((80, 12))
    cloud = build_point_cloud(
        features,
        max_points=None,
        normalize=True,
        projection="pca",
        projection_dim=5,
    )

    assert cloud.shape == (80, 5)


def test_build_point_cloud_jl_projection_reduces_dimension():
    rng = np.random.default_rng(0)
    features = rng.standard_normal((80, 12))
    cloud = build_point_cloud(
        features,
        max_points=None,
        projection="jl",
        projection_dim=6,
        projection_random_state=42,
    )

    assert cloud.shape == (80, 6)
