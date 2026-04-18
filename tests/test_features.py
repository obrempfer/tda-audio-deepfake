"""Tests for audio feature extraction."""

import numpy as np
import pytest

from tda_deepfake.features.extraction import extract_features, build_point_cloud, build_mel_spectrogram
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


def test_build_mel_spectrogram_shape_and_finiteness():
    audio = _synthetic_audio(duration_s=1.0)
    grid = build_mel_spectrogram(audio, n_mels=32, max_frames=40)

    assert grid.shape[0] == 32
    assert grid.shape[1] <= 40
    assert np.isfinite(grid).all()


@pytest.mark.parametrize("method", ["minmax", "zscore"])
def test_build_mel_spectrogram_normalization_methods(method: str):
    audio = _synthetic_audio(duration_s=0.5)
    grid = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        normalization_method=method,
    )

    assert np.isfinite(grid).all()
    if method == "minmax":
        assert np.min(grid) >= 0.0
        assert np.max(grid) <= 1.0


def test_build_mel_spectrogram_gaussian_smoothing_changes_grid():
    audio = _synthetic_audio(duration_s=0.5)
    unsmoothed = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        smoothing="none",
    )
    smoothed = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        smoothing="gaussian",
        smoothing_sigma=1.0,
    )

    assert unsmoothed.shape == smoothed.shape
    assert np.isfinite(smoothed).all()
    assert not np.allclose(unsmoothed, smoothed)


@pytest.mark.parametrize("mode,masked_slice", [
    ("drop_low", slice(0, 5)),
    ("drop_mid", slice(5, 10)),
    ("drop_high", slice(10, 16)),
])
def test_build_mel_spectrogram_band_drop_masks_expected_region(mode: str, masked_slice: slice):
    audio = _synthetic_audio(duration_s=0.5)
    grid = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="none",
        smoothing="none",
        normalize=False,
        band_mask_mode=mode,
        band_split_low=0.33,
        band_split_high=0.66,
    )

    assert np.isfinite(grid).all()
    assert np.allclose(grid[masked_slice, :], 0.0)


@pytest.mark.parametrize("mode,kept_slice", [
    ("keep_low", slice(0, 5)),
    ("keep_mid", slice(5, 10)),
    ("keep_high", slice(10, 16)),
])
def test_build_mel_spectrogram_band_keep_masks_complement(mode: str, kept_slice: slice):
    audio = _synthetic_audio(duration_s=0.5)
    grid = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="none",
        smoothing="none",
        normalize=False,
        band_mask_mode=mode,
        band_split_low=0.33,
        band_split_high=0.66,
    )

    assert np.isfinite(grid).all()
    # Kept region should preserve some nonzero energy for a sine test tone.
    assert np.any(grid[kept_slice, :] > 0.0)
    # Everything else should be fully masked.
    mask = np.ones_like(grid, dtype=bool)
    mask[kept_slice, :] = False
    assert np.allclose(grid[mask], 0.0)


@pytest.mark.parametrize("mode", ["transition", "sustained"])
def test_build_mel_spectrogram_temporal_field_modes(mode: str):
    audio = _synthetic_audio(duration_s=0.5)
    base = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="db",
        smoothing="none",
        normalize=False,
        temporal_field_mode="none",
    )
    transformed = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="db",
        smoothing="none",
        normalize=False,
        temporal_field_mode=mode,
        temporal_field_sigma=2.0,
    )

    assert np.isfinite(transformed).all()
    assert transformed.shape == base.shape
    assert not np.allclose(transformed, base)


def test_build_mel_spectrogram_energy_weighting_changes_grid():
    audio = _synthetic_audio(duration_s=0.5)
    base = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="db",
        smoothing="none",
        normalize=False,
        energy_weighting_mode="none",
    )
    weighted = build_mel_spectrogram(
        audio,
        n_mels=16,
        max_frames=20,
        compression="db",
        smoothing="none",
        normalize=False,
        energy_weighting_mode="power",
        energy_weighting_gamma=2.0,
    )

    assert np.isfinite(weighted).all()
    assert weighted.shape == base.shape
    assert not np.allclose(weighted, base)
