"""Smoke test: full pipeline on synthetic audio, no real dataset needed.

Uses the 'statistics' vectorization method so it runs with only
numpy/ripser/scikit-learn installed (no persim or giotto-tda required).
"""

import numpy as np
import pytest

from scripts.run_pipeline import _feature_cache_key, _subsample_samples
from tda_deepfake.config import FeatureConfig, PointCloudConfig, TopologyConfig, VectorizationConfig
from tda_deepfake.features.extraction import extract_features, build_point_cloud, build_mel_spectrogram
from tda_deepfake.topology.persistent_homology import compute_persistence
from tda_deepfake.topology.vectorization import vectorize_diagrams
from tda_deepfake.classification.classifier import Classifier


def _synthetic_audio(duration_s: float = 1.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration_s)).astype(np.float32)


def test_full_pipeline_smoke():
    """End-to-end: synthetic audio -> TDA feature vector -> SVM prediction."""
    n_samples = 10
    X_list, y_list = [], []

    for i in range(n_samples):
        audio = _synthetic_audio(seed=i)
        features = extract_features(
            audio,
            include_f0=False,
            include_jitter_shimmer=False,
            include_formants=False,
            include_spectral_flux=False,
        )
        cloud = build_point_cloud(features, max_points=100)
        diagrams = compute_persistence(cloud, max_dim=1)
        vec = vectorize_diagrams(diagrams, method="statistics")
        X_list.append(vec)
        y_list.append(i % 2)

    X = np.stack(X_list)
    y = np.array(y_list)

    clf = Classifier(model="svm")
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == (n_samples,)
    assert set(preds).issubset({0, 1})

    metrics = clf.evaluate(X, y)
    assert "auc" in metrics
    assert "report" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_full_cubical_pipeline_smoke():
    """End-to-end: synthetic audio -> mel grid -> cubical PH -> SVM prediction."""
    n_samples = 10
    X_list, y_list = [], []

    for i in range(n_samples):
        audio = _synthetic_audio(seed=i)
        grid = build_mel_spectrogram(audio, n_mels=24, max_frames=32)
        diagrams = compute_persistence(grid, complex_type="cubical", max_dim=1)
        vec = vectorize_diagrams(diagrams, method="statistics")
        X_list.append(vec)
        y_list.append(i % 2)

    X = np.stack(X_list)
    y = np.array(y_list)

    clf = Classifier(model="svm")
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)

    assert X.shape[0] == n_samples
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_full_knn_flag_pipeline_smoke():
    """End-to-end: synthetic audio -> point cloud -> kNN flag PH -> SVM prediction."""
    n_samples = 10
    X_list, y_list = [], []

    for i in range(n_samples):
        audio = _synthetic_audio(seed=i)
        features = extract_features(
            audio,
            include_f0=False,
            include_jitter_shimmer=False,
            include_formants=False,
            include_spectral_flux=False,
        )
        cloud = build_point_cloud(features, max_points=100)
        diagrams = compute_persistence(cloud, complex_type="knn_flag", max_dim=1, knn_k=10)
        vec = vectorize_diagrams(diagrams, method="statistics")
        X_list.append(vec)
        y_list.append(i % 2)

    X = np.stack(X_list)
    y = np.array(y_list)

    clf = Classifier(model="svm")
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)

    assert X.shape[0] == n_samples
    assert "auc" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0


def test_feature_vector_fixed_length():
    """Vectors must be same length regardless of utterance duration."""
    short_audio = _synthetic_audio(duration_s=0.5, seed=0)
    long_audio = _synthetic_audio(duration_s=2.0, seed=1)

    def _vectorize(audio):
        features = extract_features(audio, include_f0=False, include_jitter_shimmer=False,
                                     include_formants=False, include_spectral_flux=False)
        cloud = build_point_cloud(features, max_points=100)
        diagrams = compute_persistence(cloud, max_dim=1)
        return vectorize_diagrams(diagrams, method="statistics")

    v1 = _vectorize(short_audio)
    v2 = _vectorize(long_audio)
    assert v1.shape == v2.shape, f"Expected equal lengths, got {v1.shape} vs {v2.shape}"


def test_build_point_cloud_subsampling():
    """build_point_cloud must subsample to at most max_points rows."""
    rng = np.random.default_rng(0)
    big_matrix = rng.standard_normal((500, 39))
    cloud = build_point_cloud(big_matrix, max_points=200)
    assert cloud.shape == (200, 39)


def test_build_point_cloud_no_subsample_needed():
    """build_point_cloud should return the matrix unchanged when it's small enough."""
    rng = np.random.default_rng(0)
    small_matrix = rng.standard_normal((50, 39))
    cloud = build_point_cloud(small_matrix, max_points=300)
    assert cloud.shape == (50, 39)


def test_classifier_save_load(tmp_path):
    """Classifier.save() and Classifier.load() round-trip."""
    X = np.random.default_rng(0).standard_normal((20, 6))
    y = np.array([0, 1] * 10)

    clf = Classifier(model="svm")
    clf.fit(X, y)
    preds_before = clf.predict(X)

    model_path = tmp_path / "model.pkl"
    clf.save(model_path)

    clf2 = Classifier.load(model_path)
    preds_after = clf2.predict(X)

    np.testing.assert_array_equal(preds_before, preds_after)


def test_subsample_samples_preserves_both_classes():
    samples = [(f"sample_{i}", 0) for i in range(40)] + [(f"sample_{i}", 1) for i in range(40, 80)]
    subset = _subsample_samples(samples, max_samples=10, random_state=42)

    assert len(subset) == 10
    assert {label for _, label in subset} == {0, 1}


def test_feature_cache_key_changes_with_feature_flags():
    original_include_f0 = FeatureConfig.INCLUDE_F0
    try:
        FeatureConfig.INCLUDE_F0 = False
        key_without_f0 = _feature_cache_key("statistics", n_bins=20, max_points=300)

        FeatureConfig.INCLUDE_F0 = True
        key_with_f0 = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        FeatureConfig.INCLUDE_F0 = original_include_f0

    assert key_without_f0 != key_with_f0


def test_feature_cache_key_changes_with_point_cloud_preprocessing():
    original_normalize = PointCloudConfig.NORMALIZE
    try:
        PointCloudConfig.NORMALIZE = False
        key_without_normalize = _feature_cache_key("statistics", n_bins=20, max_points=300)

        PointCloudConfig.NORMALIZE = True
        key_with_normalize = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        PointCloudConfig.NORMALIZE = original_normalize

    assert key_without_normalize != key_with_normalize


def test_feature_cache_key_changes_with_landscape_config():
    original_layers = VectorizationConfig.LANDSCAPE_N_LAYERS
    try:
        VectorizationConfig.LANDSCAPE_N_LAYERS = 5
        key_default = _feature_cache_key("landscape", n_bins=20, max_points=300)

        VectorizationConfig.LANDSCAPE_N_LAYERS = 7
        key_changed = _feature_cache_key("landscape", n_bins=20, max_points=300)
    finally:
        VectorizationConfig.LANDSCAPE_N_LAYERS = original_layers

    assert key_default != key_changed


def test_feature_cache_key_changes_with_complex_type():
    original_complex = TopologyConfig.COMPLEX
    try:
        TopologyConfig.COMPLEX = "vietoris_rips"
        key_vr = _feature_cache_key("statistics", n_bins=20, max_points=300)

        TopologyConfig.COMPLEX = "cubical"
        key_cubical = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        TopologyConfig.COMPLEX = original_complex

    assert key_vr != key_cubical


def test_feature_cache_key_changes_with_knn_graph_parameters():
    original_k = TopologyConfig.KNN_K
    try:
        TopologyConfig.KNN_K = 10
        key_k10 = _feature_cache_key("statistics", n_bins=20, max_points=300)

        TopologyConfig.KNN_K = 20
        key_k20 = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        TopologyConfig.KNN_K = original_k

    assert key_k10 != key_k20
