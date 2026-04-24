"""Smoke test: full pipeline on synthetic audio, no real dataset needed.

Uses the 'statistics' vectorization method so it runs with only
numpy/ripser/scikit-learn installed (no persim or giotto-tda required).
"""

import wave

import numpy as np
import pytest

from scripts.run_pipeline import (
    _diagram_cache_key,
    _extract_split,
    _feature_cache_key,
    _resolve_worker_count,
    _subsample_samples,
    _vector_block_cache_key,
)
from tda_deepfake.config import (
    FeatureConfig,
    PointCloudConfig,
    SpectrogramConfig,
    TopologyConfig,
    VectorizationConfig,
    apply_runtime_config,
    export_runtime_config,
)
from tda_deepfake.features.extraction import (
    build_mel_spectrogram,
    build_point_cloud,
    build_raw_mel_spectrogram,
    extract_features,
    postprocess_mel_spectrogram,
)
from tda_deepfake.topology.persistent_homology import compute_persistence
from tda_deepfake.topology.morse_smale import compute_morse_smale_signature
from tda_deepfake.topology.vectorization import (
    flatten_vector_blocks,
    vectorize_diagram_blocks,
    vectorize_diagrams,
)
from tda_deepfake.classification.classifier import Classifier


def _synthetic_audio(duration_s: float = 1.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration_s)).astype(np.float32)


def _write_wav(path, audio: np.ndarray, sr: int = 16000) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())


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
    assert "eer" in metrics
    assert "report" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0


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
    assert "eer" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0


def test_mel_staging_round_trip_matches_direct_builder():
    audio = _synthetic_audio(seed=123)
    direct = build_mel_spectrogram(
        audio,
        n_mels=24,
        band_mask_mode="keep_low",
        compression="log1p",
        smoothing="gaussian",
        smoothing_sigma=0.5,
        max_frames=32,
    )
    raw = build_raw_mel_spectrogram(audio, n_mels=24)
    staged = postprocess_mel_spectrogram(
        raw,
        band_mask_mode="keep_low",
        compression="log1p",
        smoothing="gaussian",
        smoothing_sigma=0.5,
        max_frames=32,
    )
    np.testing.assert_allclose(direct, staged)


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
    assert "eer" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0


def test_full_morse_smale_pipeline_smoke():
    """End-to-end: synthetic audio -> mel grid -> Morse-Smale-inspired signature -> SVM."""
    n_samples = 10
    X_list, y_list = [], []

    for i in range(n_samples):
        audio = _synthetic_audio(seed=i)
        grid = build_mel_spectrogram(audio, n_mels=24, max_frames=32)
        vec = compute_morse_smale_signature(grid, neighborhood_size=3, top_k_basins=4, top_k_extrema=4)
        X_list.append(vec)
        y_list.append(i % 2)

    X = np.stack(X_list)
    y = np.array(y_list)

    clf = Classifier(model="svm")
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)

    assert X.shape[0] == n_samples
    assert "auc" in metrics
    assert "eer" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0


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


def test_vector_block_round_trip_matches_flat_vectorization():
    audio = _synthetic_audio(seed=7)
    grid = build_mel_spectrogram(audio, n_mels=24, max_frames=32)
    diagrams = compute_persistence(grid, complex_type="cubical", max_dim=1)

    direct = vectorize_diagrams(diagrams, method="statistics")
    blocks = vectorize_diagram_blocks(diagrams, method="statistics")
    staged = flatten_vector_blocks(blocks)

    np.testing.assert_allclose(direct, staged)


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


def test_runtime_config_snapshot_round_trip():
    original = FeatureConfig.INCLUDE_F0
    snapshot = export_runtime_config()
    try:
        FeatureConfig.INCLUDE_F0 = not original
        apply_runtime_config(snapshot)
        assert FeatureConfig.INCLUDE_F0 == original
    finally:
        FeatureConfig.INCLUDE_F0 = original


def test_resolve_worker_count_auto_clamps_to_sample_count():
    assert _resolve_worker_count(0, 3) == 3
    assert _resolve_worker_count(8, 3) == 3
    assert _resolve_worker_count(1, 3) == 1


def test_extract_split_parallel_matches_serial(tmp_path):
    samples = []
    for i in range(4):
        audio_path = tmp_path / f"sample_{i}.wav"
        _write_wav(audio_path, _synthetic_audio(duration_s=0.1, seed=i))
        samples.append((audio_path, i % 2))

    X_serial, y_serial = _extract_split(
        samples,
        tmp_path / "cache_serial",
        method="statistics",
        n_bins=20,
        max_points=50,
        num_workers=1,
        progress_every=10,
    )
    X_parallel, y_parallel = _extract_split(
        samples,
        tmp_path / "cache_parallel",
        method="statistics",
        n_bins=20,
        max_points=50,
        num_workers=2,
        progress_every=10,
    )

    np.testing.assert_allclose(X_serial, X_parallel)
    np.testing.assert_array_equal(y_serial, y_parallel)


def test_diagram_cache_key_ignores_requested_homology_dim():
    original_max_dim = TopologyConfig.MAX_HOMOLOGY_DIM
    try:
        TopologyConfig.MAX_HOMOLOGY_DIM = 0
        key_h0 = _diagram_cache_key("processed-grid")

        TopologyConfig.MAX_HOMOLOGY_DIM = 1
        key_h1 = _diagram_cache_key("processed-grid")
    finally:
        TopologyConfig.MAX_HOMOLOGY_DIM = original_max_dim

    assert key_h0 == key_h1


def test_vector_block_cache_key_ignores_homology_weights():
    original_weights = VectorizationConfig.HOMOLOGY_WEIGHTS
    try:
        VectorizationConfig.HOMOLOGY_WEIGHTS = None
        unweighted_key = _vector_block_cache_key("diagram-key", method="statistics", n_bins=20)

        VectorizationConfig.HOMOLOGY_WEIGHTS = [0.0, 1.0]
        weighted_key = _vector_block_cache_key("diagram-key", method="statistics", n_bins=20)
        final_weighted = _feature_cache_key("statistics", n_bins=20, max_points=300)

        VectorizationConfig.HOMOLOGY_WEIGHTS = None
        final_unweighted = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        VectorizationConfig.HOMOLOGY_WEIGHTS = original_weights

    assert unweighted_key == weighted_key
    assert final_unweighted != final_weighted


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


def test_feature_cache_key_changes_with_spectrogram_smoothing():
    original_smoothing = SpectrogramConfig.SMOOTHING
    try:
        SpectrogramConfig.SMOOTHING = "none"
        key_unsmoothed = _feature_cache_key("landscape", n_bins=20, max_points=300)

        SpectrogramConfig.SMOOTHING = "gaussian"
        key_smoothed = _feature_cache_key("landscape", n_bins=20, max_points=300)
    finally:
        SpectrogramConfig.SMOOTHING = original_smoothing

    assert key_unsmoothed != key_smoothed


def test_feature_cache_key_changes_with_morse_smale_config():
    from tda_deepfake.config import MorseSmaleConfig

    original_top_k = MorseSmaleConfig.TOP_K_BASINS
    try:
        MorseSmaleConfig.TOP_K_BASINS = 4
        key_small = _feature_cache_key("statistics", n_bins=20, max_points=300)

        MorseSmaleConfig.TOP_K_BASINS = 8
        key_large = _feature_cache_key("statistics", n_bins=20, max_points=300)
    finally:
        MorseSmaleConfig.TOP_K_BASINS = original_top_k

    assert key_small != key_large
