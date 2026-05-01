import wave

import numpy as np

from scripts.run_pipeline import _extract_split
from tda_deepfake.classification.classifier import Classifier
from tda_deepfake.config import (
    PointCloudConfig,
    SpectrogramConfig,
    TakensConfig,
    TopologyConfig,
    apply_runtime_config,
    export_runtime_config,
)
from tda_deepfake.features import build_point_cloud
from tda_deepfake.topology import build_takens_embedding, build_takens_signal, compute_persistence
from tda_deepfake.topology.vectorization import vectorize_diagrams


def _synthetic_audio(duration_s: float = 1.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, duration_s, int(sr * duration_s), endpoint=False)
    carrier = 0.25 * np.sin(2.0 * np.pi * (220.0 + 10.0 * seed) * time)
    noise = 0.05 * rng.standard_normal(time.shape[0])
    return np.asarray(carrier + noise, dtype=np.float32)


def _write_wav(path, audio: np.ndarray, sr: int = 16000) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm.tobytes())


def test_takens_signal_shapes():
    audio = _synthetic_audio(duration_s=1.0, seed=7)

    low_wave = build_takens_signal(audio, signal_type="low_wave")
    full_wave = build_takens_signal(audio, signal_type="full_wave")
    low_env = build_takens_signal(audio, signal_type="low_env")
    full_env = build_takens_signal(audio, signal_type="full_env")

    assert low_wave.ndim == 1
    assert full_wave.ndim == 1
    assert low_env.ndim == 1
    assert full_env.ndim == 1
    assert low_wave.shape == full_wave.shape
    assert low_env.shape == full_env.shape
    assert 0 < low_env.shape[0] < low_wave.shape[0]


def test_takens_embedding_correctness():
    signal = np.arange(10, dtype=np.float64)
    embedding = build_takens_embedding(signal, embedding_dim=3, delay=2, stride=1)
    expected = np.array(
        [
            [0.0, 2.0, 4.0],
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
            [3.0, 5.0, 7.0],
            [4.0, 6.0, 8.0],
            [5.0, 7.0, 9.0],
        ]
    )
    np.testing.assert_allclose(embedding, expected)


def test_full_takens_pipeline_smoke():
    n_samples = 10
    X_list, y_list = [], []

    for i in range(n_samples):
        audio = _synthetic_audio(seed=i)
        signal = build_takens_signal(
            audio,
            signal_type="low_env",
            envelope_smooth_sigma=1.0,
        )
        embedding = build_takens_embedding(signal, embedding_dim=3, delay=1, stride=1)
        point_cloud = build_point_cloud(embedding, max_points=64)
        diagrams = compute_persistence(point_cloud, complex_type="vietoris_rips", max_dim=1)
        vec = vectorize_diagrams(diagrams, method="statistics")
        X_list.append(vec)
        y_list.append(i % 2)

    X = np.stack(X_list)
    y = np.array(y_list)

    clf = Classifier(model="svm")
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)

    assert X.shape[0] == n_samples
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0


def test_takens_extract_split_smoke(tmp_path):
    snapshot = export_runtime_config()
    try:
        apply_runtime_config(
            {
                "point_cloud": {
                    "normalize": False,
                    "normalization_method": PointCloudConfig.NORMALIZATION_METHOD,
                    "projection": "none",
                    "projection_dim": None,
                    "projection_random_state": PointCloudConfig.PROJECTION_RANDOM_STATE,
                },
                "spectrogram": {
                    "n_mels": 24,
                    "power": SpectrogramConfig.POWER,
                    "fmin": SpectrogramConfig.FMIN,
                    "fmax": SpectrogramConfig.FMAX,
                    "band_split_low": 0.33,
                },
                "takens": {
                    "signal_type": "low_env",
                    "lowpass_cutoff_hz": TakensConfig.LOWPASS_CUTOFF_HZ,
                    "filter_order": TakensConfig.FILTER_ORDER,
                    "signal_normalization": "zscore",
                    "envelope_compression": "log1p",
                    "envelope_smooth_sigma": 1.0,
                    "embedding_dim": 3,
                    "delay": 1,
                    "stride": 1,
                    "max_points": 64,
                },
                "topology": {
                    "complex": "takens_ph",
                    "max_homology_dim": 1,
                    "distance_metric": "euclidean",
                    "coeff": 2,
                },
            }
        )

        audio_dir = tmp_path / "audio"
        cache_dir = tmp_path / "cache"
        audio_dir.mkdir()

        samples = []
        for i in range(6):
            path = audio_dir / f"utt_{i}.wav"
            _write_wav(path, _synthetic_audio(seed=i))
            samples.append((path, i % 2))

        X, y = _extract_split(
            samples,
            cache_dir=cache_dir,
            method="statistics",
            n_bins=10,
            max_points=64,
            num_workers=1,
            progress_every=0,
        )

        assert X.shape[0] == len(samples)
        assert y.shape == (len(samples),)
        assert np.isfinite(X).all()
    finally:
        apply_runtime_config(snapshot)
