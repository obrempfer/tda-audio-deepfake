"""CLI entry point: run the full TDA detection pipeline.

Usage (CV-only mode — quick experiments on a single split):
    python -m scripts.run_pipeline \
        --protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
        --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
        --out-dir data/results/run_01 \
        --method statistics --model svm --max-samples 500

Usage (train/eval mode — full experiment):
    python -m scripts.run_pipeline \
        --train-protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
        --train-audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
        --eval-protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt \
        --eval-audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac \
        --out-dir data/results/baseline_dev \
        --method persistence_image --n-bins 20 --model svm

Feature vectors are cached per utterance under <out-dir>/feature_cache/.
Re-runs with the same --method and --n-bins skip extraction entirely.
"""

import argparse
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from pathlib import Path
from typing import Callable

from tda_deepfake.utils import load_audio, load_asvspoof_manifest
from tda_deepfake.features import (
    build_point_cloud,
    build_raw_mel_spectrogram,
    extract_features,
    postprocess_mel_spectrogram,
)
from tda_deepfake.topology import (
    compute_morse_smale_signature,
    compute_persistence,
    flatten_vector_blocks,
    vectorize_diagram_blocks,
)
from tda_deepfake.classification import Classifier
from tda_deepfake.config import (
    AudioConfig,
    ClassifierConfig,
    FeatureConfig,
    PointCloudConfig,
    MorseSmaleConfig,
    SpectrogramConfig,
    TopologyConfig,
    VectorizationConfig,
    apply_runtime_config,
    export_runtime_config,
    load_config_from_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TDA Audio Deepfake Detection Pipeline")

    # CV-only mode
    parser.add_argument("--protocol", type=Path, default=None,
                        help="ASVspoof protocol file (CV-only mode)")
    parser.add_argument("--audio-dir", type=Path, default=None,
                        help="Directory of .flac files (CV-only mode)")

    # Train/eval mode
    parser.add_argument("--train-protocol", type=Path, default=None,
                        help="Train protocol file (enables train/eval mode)")
    parser.add_argument("--train-audio-dir", type=Path, default=None)
    parser.add_argument("--eval-protocol", type=Path, default=None,
                        help="Eval/dev protocol file")
    parser.add_argument("--eval-audio-dir", type=Path, default=None)

    # Shared options
    parser.add_argument("--out-dir", type=Path, default=Path("data/results/default"),
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Legacy cap on train/CV samples (useful for quick smoke tests)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Cap on train samples in train/eval mode. Overrides --max-samples.")
    parser.add_argument("--max-eval-samples", type=int, default=None,
                        help="Cap on eval samples in train/eval mode.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Feature cache directory for CV mode, or parent cache directory for "
                             "train/eval mode. If omitted, uses <out-dir>/feature_cache.")
    parser.add_argument("--train-cache-dir", type=Path, default=None,
                        help="Feature cache directory for train split in train/eval mode.")
    parser.add_argument("--eval-cache-dir", type=Path, default=None,
                        help="Feature cache directory for eval split in train/eval mode.")
    parser.add_argument("--load-model", type=Path, default=None,
                        help="Load a saved model.pkl and skip train extraction/training in train/eval mode.")
    parser.add_argument("--method", default=None,
                        choices=["persistence_image", "landscape", "statistics"],
                        help="Vectorization method")
    parser.add_argument("--model", default=None, choices=["svm", "logistic"],
                        help="Classifier type")
    parser.add_argument("--n-bins", type=int, default=None,
                        help="Persistence image grid resolution (n_bins x n_bins)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max point cloud size per utterance (subsampled uniformly)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Feature-extraction worker processes. Use 0 to use all visible CPUs.")
    parser.add_argument("--train-workers", type=int, default=None,
                        help="Override --num-workers for train extraction in train/eval mode.")
    parser.add_argument("--eval-workers", type=int, default=None,
                        help="Override --num-workers for eval extraction in train/eval mode.")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="Print extraction progress every N completed samples.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config file to override defaults")
    parser.add_argument("--ablation", action="store_true",
                        help="Run dimensional ablation on flagged samples (not yet wired)")
    return parser.parse_args()


_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_THREADPOOL_LIMITS = None


def _limit_worker_threads() -> None:
    """Avoid nested BLAS/OpenMP oversubscription inside worker processes."""
    global _THREADPOOL_LIMITS

    for key in _THREAD_ENV_VARS:
        os.environ[key] = "1"

    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMITS = threadpool_limits(limits=1)
        _THREADPOOL_LIMITS.__enter__()
    except Exception:
        _THREADPOOL_LIMITS = None


_STAGE_CACHE_ROOT = "_stage_cache"


def _stable_digest(signature: dict[str, object]) -> str:
    """Hash a config signature into a short stable cache suffix."""
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _stage_cache_file(
    cache_dir: Path,
    stage_name: str,
    audio_path: Path,
    cache_key: str,
    suffix: str = ".npy",
) -> Path:
    """Return a stage-cache path nested under the split cache directory."""
    stage_dir = cache_dir / _STAGE_CACHE_ROOT / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    return stage_dir / f"{audio_path.stem}_{cache_key}{suffix}"


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    """Atomically save one ndarray to avoid partially-written cache files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.save(tmp_path, array)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_save_bundle(path: Path, arrays: list[np.ndarray], computed_max_dim: int) -> None:
    """Atomically save a variable-length list of arrays plus bundle metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "meta_computed_max_dim": np.asarray(computed_max_dim, dtype=np.int64),
    }
    for idx, array in enumerate(arrays):
        payload[f"arr_{idx}"] = np.asarray(array, dtype=np.float64)

    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_cached_array(path: Path) -> np.ndarray | None:
    """Load one cached ndarray, dropping corrupt files eagerly."""
    if not path.exists():
        return None
    try:
        return np.load(path, allow_pickle=False)
    except Exception:
        path.unlink(missing_ok=True)
        return None


def _load_cached_bundle(path: Path) -> tuple[list[np.ndarray], int] | None:
    """Load a cached list-of-arrays bundle, or None if absent/corrupt."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            computed_max_dim = int(data["meta_computed_max_dim"][()])
            arrays = []
            idx = 0
            while f"arr_{idx}" in data.files:
                arrays.append(np.asarray(data[f"arr_{idx}"], dtype=np.float64))
                idx += 1
    except Exception:
        path.unlink(missing_ok=True)
        return None

    return arrays, computed_max_dim


def _raw_mel_cache_key() -> str:
    """Cache key for the raw mel grid before masking/compression."""
    return _stable_digest(
        {
            "stage": "raw_mel",
            "audio": {
                "sample_rate": AudioConfig.SAMPLE_RATE,
                "window_size_ms": AudioConfig.WINDOW_SIZE_MS,
                "hop_size_ms": AudioConfig.HOP_SIZE_MS,
            },
            "spectrogram": {
                "kind": SpectrogramConfig.KIND,
                "n_mels": SpectrogramConfig.N_MELS,
                "power": SpectrogramConfig.POWER,
                "fmin": SpectrogramConfig.FMIN,
                "fmax": SpectrogramConfig.FMAX,
            },
        }
    )


def _processed_grid_cache_key(raw_mel_key: str) -> str:
    """Cache key for the fully processed cubical grid."""
    return _stable_digest(
        {
            "stage": "processed_grid",
            "raw_mel_key": raw_mel_key,
            "spectrogram": {
                "log_scale": SpectrogramConfig.LOG_SCALE,
                "compression": SpectrogramConfig.COMPRESSION,
                "smoothing": SpectrogramConfig.SMOOTHING,
                "smoothing_sigma": SpectrogramConfig.SMOOTHING_SIGMA,
                "smoothing_axis": SpectrogramConfig.SMOOTHING_AXIS,
                "band_mask_mode": SpectrogramConfig.BAND_MASK_MODE,
                "band_split_low": SpectrogramConfig.BAND_SPLIT_LOW,
                "band_split_high": SpectrogramConfig.BAND_SPLIT_HIGH,
                "band_mask_fill": SpectrogramConfig.BAND_MASK_FILL,
                "temporal_field_mode": SpectrogramConfig.TEMPORAL_FIELD_MODE,
                "temporal_field_sigma": SpectrogramConfig.TEMPORAL_FIELD_SIGMA,
                "energy_weighting_mode": SpectrogramConfig.ENERGY_WEIGHTING_MODE,
                "energy_weighting_gamma": SpectrogramConfig.ENERGY_WEIGHTING_GAMMA,
                "energy_gate_percentile": SpectrogramConfig.ENERGY_GATE_PERCENTILE,
                "energy_gate_fill": SpectrogramConfig.ENERGY_GATE_FILL,
                "normalize": SpectrogramConfig.NORMALIZE,
                "normalization_method": SpectrogramConfig.NORMALIZATION_METHOD,
                "max_frames": SpectrogramConfig.MAX_FRAMES,
            },
        }
    )


def _feature_matrix_cache_key() -> str:
    """Cache key for the point-cloud feature matrix before subsampling."""
    return _stable_digest(
        {
            "stage": "feature_matrix",
            "audio": {
                "sample_rate": AudioConfig.SAMPLE_RATE,
                "window_size_ms": AudioConfig.WINDOW_SIZE_MS,
                "hop_size_ms": AudioConfig.HOP_SIZE_MS,
                "n_mfcc": AudioConfig.N_MFCC,
                "include_delta": AudioConfig.INCLUDE_DELTA,
                "include_delta2": AudioConfig.INCLUDE_DELTA2,
            },
            "feature": {
                "include_f0": FeatureConfig.INCLUDE_F0,
                "include_jitter_shimmer": FeatureConfig.INCLUDE_JITTER_SHIMMER,
                "include_formants": FeatureConfig.INCLUDE_FORMANTS,
                "include_spectral_flux": FeatureConfig.INCLUDE_SPECTRAL_FLUX,
                "n_formants": FeatureConfig.N_FORMANTS,
            },
        }
    )


def _point_cloud_cache_key(feature_matrix_key: str, max_points: int) -> str:
    """Cache key for the normalized/projected point cloud."""
    return _stable_digest(
        {
            "stage": "point_cloud",
            "feature_matrix_key": feature_matrix_key,
            "point_cloud": {
                "max_points": max_points,
                "normalize": PointCloudConfig.NORMALIZE,
                "normalization_method": PointCloudConfig.NORMALIZATION_METHOD,
                "projection": PointCloudConfig.PROJECTION,
                "projection_dim": PointCloudConfig.PROJECTION_DIM,
                "projection_random_state": PointCloudConfig.PROJECTION_RANDOM_STATE,
            },
        }
    )


def _diagram_cache_key(topological_object_key: str) -> str:
    """Cache key for persistence diagrams, excluding the requested max dimension."""
    return _stable_digest(
        {
            "stage": "diagrams",
            "topological_object_key": topological_object_key,
            "topology": {
                "complex": TopologyConfig.COMPLEX,
                "distance_metric": TopologyConfig.DISTANCE_METRIC,
                "cubical_filtration": TopologyConfig.CUBICAL_FILTRATION,
                "knn_k": TopologyConfig.KNN_K,
                "knn_graph_mode": TopologyConfig.KNN_GRAPH_MODE,
                "max_edge_length": TopologyConfig.MAX_EDGE_LENGTH,
                "coeff": TopologyConfig.COEFF,
            },
        }
    )


def _vector_block_cache_key(diagram_key: str, method: str, n_bins: int) -> str:
    """Cache key for per-dimension vector blocks, excluding homology weights."""
    return _stable_digest(
        {
            "stage": "vector_blocks",
            "diagram_key": diagram_key,
            "vectorization": {
                "method": method,
                "n_bins": n_bins,
                "sigma": VectorizationConfig.PI_SIGMA,
                "landscape_n_layers": VectorizationConfig.LANDSCAPE_N_LAYERS,
                "landscape_n_bins": VectorizationConfig.LANDSCAPE_N_BINS,
            },
        }
    )


def _morse_smale_signature_cache_key(processed_grid_key: str) -> str:
    """Cache key for topology-inspired Morse-Smale signatures."""
    return _stable_digest(
        {
            "stage": "morse_smale_signature",
            "processed_grid_key": processed_grid_key,
            "complex": TopologyConfig.COMPLEX,
            "morse_smale": {
                "implementation": MorseSmaleConfig.IMPLEMENTATION,
                "graph_max_neighbors": MorseSmaleConfig.GRAPH_MAX_NEIGHBORS,
                "graph_relaxed": MorseSmaleConfig.GRAPH_RELAXED,
                "normalization": MorseSmaleConfig.NORMALIZATION,
                "simplification": MorseSmaleConfig.SIMPLIFICATION,
                "neighborhood_size": MorseSmaleConfig.NEIGHBORHOOD_SIZE,
                "top_k_basins": MorseSmaleConfig.TOP_K_BASINS,
                "include_extrema_values": MorseSmaleConfig.INCLUDE_EXTREMA_VALUES,
                "top_k_extrema": MorseSmaleConfig.TOP_K_EXTREMA,
            },
        }
    )


def _load_or_compute_stage_array(
    cache_file: Path,
    compute_fn: Callable[[], np.ndarray],
) -> np.ndarray:
    """Read a stage cache file or compute and atomically materialize it."""
    cached = _load_cached_array(cache_file)
    if cached is not None:
        return cached

    array = np.asarray(compute_fn(), dtype=np.float64)
    _atomic_save_array(cache_file, array)
    return array


def _load_or_compute_dimensional_bundle(
    cache_file: Path,
    requested_max_dim: int,
    compute_fn: Callable[[int], list[np.ndarray]],
) -> list[np.ndarray]:
    """Load a staged bundle if it covers the requested dimensions, else recompute."""
    cached = _load_cached_bundle(cache_file)
    if cached is not None:
        arrays, computed_max_dim = cached
        if computed_max_dim >= requested_max_dim and len(arrays) >= requested_max_dim + 1:
            return arrays[: requested_max_dim + 1]

    arrays = [np.asarray(array, dtype=np.float64) for array in compute_fn(requested_max_dim)]
    _atomic_save_bundle(cache_file, arrays, computed_max_dim=requested_max_dim)
    return arrays


def _compute_feature_vector(audio_path: Path, cache_dir: Path, method: str, n_bins: int, max_points: int) -> np.ndarray:
    """Compute one utterance feature vector, reusing staged intermediate caches."""
    audio: np.ndarray | None = None

    def _audio() -> np.ndarray:
        nonlocal audio
        if audio is None:
            audio = load_audio(audio_path, sample_rate=AudioConfig.SAMPLE_RATE)
        return audio

    if TopologyConfig.COMPLEX in {"cubical", "morse_smale", "morse_smale_approx"}:
        raw_mel_key = _raw_mel_cache_key()
        raw_mel_file = _stage_cache_file(cache_dir, "raw_mel", audio_path, raw_mel_key)
        raw_grid = _load_or_compute_stage_array(
            raw_mel_file,
            lambda: build_raw_mel_spectrogram(
                _audio(),
                sample_rate=AudioConfig.SAMPLE_RATE,
                n_mels=SpectrogramConfig.N_MELS,
                power=SpectrogramConfig.POWER,
                fmin=SpectrogramConfig.FMIN,
                fmax=SpectrogramConfig.FMAX,
            ),
        )

        processed_grid_key = _processed_grid_cache_key(raw_mel_key)
        processed_grid_file = _stage_cache_file(cache_dir, "processed_grid", audio_path, processed_grid_key)
        grid = _load_or_compute_stage_array(
            processed_grid_file,
            lambda: postprocess_mel_spectrogram(
                raw_grid,
                log_scale=SpectrogramConfig.LOG_SCALE,
                compression=SpectrogramConfig.COMPRESSION,
                smoothing=SpectrogramConfig.SMOOTHING,
                smoothing_sigma=SpectrogramConfig.SMOOTHING_SIGMA,
                smoothing_axis=SpectrogramConfig.SMOOTHING_AXIS,
                band_mask_mode=SpectrogramConfig.BAND_MASK_MODE,
                band_split_low=SpectrogramConfig.BAND_SPLIT_LOW,
                band_split_high=SpectrogramConfig.BAND_SPLIT_HIGH,
                band_mask_fill=SpectrogramConfig.BAND_MASK_FILL,
                temporal_field_mode=SpectrogramConfig.TEMPORAL_FIELD_MODE,
                temporal_field_sigma=SpectrogramConfig.TEMPORAL_FIELD_SIGMA,
                energy_weighting_mode=SpectrogramConfig.ENERGY_WEIGHTING_MODE,
                energy_weighting_gamma=SpectrogramConfig.ENERGY_WEIGHTING_GAMMA,
                energy_gate_percentile=SpectrogramConfig.ENERGY_GATE_PERCENTILE,
                energy_gate_fill=SpectrogramConfig.ENERGY_GATE_FILL,
                normalize=SpectrogramConfig.NORMALIZE,
                normalization_method=SpectrogramConfig.NORMALIZATION_METHOD,
                max_frames=SpectrogramConfig.MAX_FRAMES,
            ),
        )

        if TopologyConfig.COMPLEX in {"morse_smale", "morse_smale_approx"}:
            signature_key = _morse_smale_signature_cache_key(processed_grid_key)
            signature_file = _stage_cache_file(cache_dir, "morse_smale_signature", audio_path, signature_key)
            return _load_or_compute_stage_array(
                signature_file,
                lambda: compute_morse_smale_signature(
                    grid,
                    implementation="approx"
                    if TopologyConfig.COMPLEX == "morse_smale_approx"
                    else MorseSmaleConfig.IMPLEMENTATION,
                    graph_max_neighbors=MorseSmaleConfig.GRAPH_MAX_NEIGHBORS,
                    graph_relaxed=MorseSmaleConfig.GRAPH_RELAXED,
                    normalization=MorseSmaleConfig.NORMALIZATION,
                    simplification=MorseSmaleConfig.SIMPLIFICATION,
                    neighborhood_size=MorseSmaleConfig.NEIGHBORHOOD_SIZE,
                    top_k_basins=MorseSmaleConfig.TOP_K_BASINS,
                    include_extrema_values=MorseSmaleConfig.INCLUDE_EXTREMA_VALUES,
                    top_k_extrema=MorseSmaleConfig.TOP_K_EXTREMA,
                ),
            )

        topology_object = grid
        topology_object_key = processed_grid_key
    else:
        feature_matrix_key = _feature_matrix_cache_key()
        feature_matrix_file = _stage_cache_file(cache_dir, "feature_matrix", audio_path, feature_matrix_key)
        feature_matrix = _load_or_compute_stage_array(
            feature_matrix_file,
            lambda: extract_features(
                _audio(),
                sample_rate=AudioConfig.SAMPLE_RATE,
                include_delta=AudioConfig.INCLUDE_DELTA,
                include_delta2=AudioConfig.INCLUDE_DELTA2,
                include_f0=FeatureConfig.INCLUDE_F0,
                include_jitter_shimmer=FeatureConfig.INCLUDE_JITTER_SHIMMER,
                include_formants=FeatureConfig.INCLUDE_FORMANTS,
                include_spectral_flux=FeatureConfig.INCLUDE_SPECTRAL_FLUX,
            ),
        )

        point_cloud_key = _point_cloud_cache_key(feature_matrix_key, max_points)
        point_cloud_file = _stage_cache_file(cache_dir, "point_cloud", audio_path, point_cloud_key)
        topology_object = _load_or_compute_stage_array(
            point_cloud_file,
            lambda: build_point_cloud(
                feature_matrix,
                max_points=max_points,
                normalize=PointCloudConfig.NORMALIZE,
                normalization_method=PointCloudConfig.NORMALIZATION_METHOD,
                projection=PointCloudConfig.PROJECTION,
                projection_dim=PointCloudConfig.PROJECTION_DIM,
                projection_random_state=PointCloudConfig.PROJECTION_RANDOM_STATE,
            ),
        )
        topology_object_key = point_cloud_key

    diagram_key = _diagram_cache_key(topology_object_key)
    diagram_file = _stage_cache_file(cache_dir, "diagrams", audio_path, diagram_key, suffix=".npz")
    diagrams = _load_or_compute_dimensional_bundle(
        diagram_file,
        requested_max_dim=TopologyConfig.MAX_HOMOLOGY_DIM,
        compute_fn=lambda max_dim: compute_persistence(
            topology_object,
            complex_type=TopologyConfig.COMPLEX,
            max_dim=max_dim,
            metric=TopologyConfig.DISTANCE_METRIC,
            cubical_filtration=TopologyConfig.CUBICAL_FILTRATION,
            knn_k=TopologyConfig.KNN_K,
            knn_graph_mode=TopologyConfig.KNN_GRAPH_MODE,
            max_edge_length=TopologyConfig.MAX_EDGE_LENGTH,
            coeff=TopologyConfig.COEFF,
        ),
    )

    block_key = _vector_block_cache_key(diagram_key, method=method, n_bins=n_bins)
    block_file = _stage_cache_file(cache_dir, "vector_blocks", audio_path, block_key, suffix=".npz")
    blocks = _load_or_compute_dimensional_bundle(
        block_file,
        requested_max_dim=TopologyConfig.MAX_HOMOLOGY_DIM,
        compute_fn=lambda _: vectorize_diagram_blocks(
            diagrams,
            method=method,
            n_bins=n_bins,
            sigma=VectorizationConfig.PI_SIGMA,
        ),
    )
    return flatten_vector_blocks(blocks, homology_weights=VectorizationConfig.HOMOLOGY_WEIGHTS)


def _load_or_compute_feature_vector(
    audio_path: Path,
    cache_dir: Path,
    cache_key: str,
    method: str,
    n_bins: int,
    max_points: int,
) -> np.ndarray:
    """Load a cached vector if present, otherwise compute it via staged caches."""
    cache_file = cache_dir / f"{audio_path.stem}_{cache_key}.npy"
    cached = _load_cached_array(cache_file)
    if cached is not None:
        return cached

    vec = _compute_feature_vector(
        audio_path,
        cache_dir=cache_dir,
        method=method,
        n_bins=n_bins,
        max_points=max_points,
    )
    _atomic_save_array(cache_file, vec)
    return vec


def _init_extraction_worker(config_snapshot: dict[str, dict[str, object]]) -> None:
    """Reapply runtime config in child processes before extraction begins."""
    apply_runtime_config(config_snapshot)
    _limit_worker_threads()


def _extract_one_sample(
    task: tuple[int, str, int, str, str, str, int, int]
) -> tuple[int, np.ndarray, int]:
    """Process one sample task in a worker process."""
    idx, audio_path_str, label, cache_dir_str, cache_key, method, n_bins, max_points = task
    audio_path = Path(audio_path_str)
    cache_dir = Path(cache_dir_str)
    try:
        vec = _load_or_compute_feature_vector(
            audio_path,
            cache_dir=cache_dir,
            cache_key=cache_key,
            method=method,
            n_bins=n_bins,
            max_points=max_points,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to extract features for {audio_path}") from exc
    return idx, vec, label


def _resolve_worker_count(requested_workers: int | None, n_samples: int) -> int:
    """Normalize worker count; 0 means all visible CPUs."""
    if n_samples <= 0:
        return 1
    if requested_workers is None:
        requested_workers = 1
    if requested_workers < 0:
        raise ValueError("Worker count must be >= 0")
    if requested_workers == 0:
        requested_workers = os.cpu_count() or 1
    return max(1, min(requested_workers, n_samples))


def _parallel_chunksize(n_samples: int, worker_count: int) -> int:
    """Choose a coarse chunk size to reduce executor overhead."""
    if worker_count <= 1:
        return 1
    return max(1, min(64, n_samples // (worker_count * 8)))


def _extract_split(
    samples: list,
    cache_dir: Path,
    method: str,
    n_bins: int,
    max_points: int,
    num_workers: int = 1,
    progress_every: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (or load from cache) feature vectors for a list of samples."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    cache_key = _feature_cache_key(method, n_bins, max_points)
    worker_count = _resolve_worker_count(num_workers, total)

    if total == 0:
        return np.zeros((0, 0), dtype=np.float64), np.array([], dtype=int)

    if worker_count == 1:
        X_list, y_list = [], []
        for i, (audio_path, label) in enumerate(samples):
            if progress_every > 0 and i % progress_every == 0:
                print(f"  [{i}/{total}] {audio_path.name}")
            vec = _load_or_compute_feature_vector(
                audio_path,
                cache_dir=cache_dir,
                cache_key=cache_key,
                method=method,
                n_bins=n_bins,
                max_points=max_points,
            )
            X_list.append(vec)
            y_list.append(label)
        return np.stack(X_list), np.array(y_list)

    print(f"Using {worker_count} worker processes for extraction")
    tasks = [
        (i, str(audio_path), int(label), str(cache_dir), cache_key, method, n_bins, max_points)
        for i, (audio_path, label) in enumerate(samples)
    ]
    vectors: list[np.ndarray | None] = [None] * total
    labels = np.empty(total, dtype=int)
    chunksize = _parallel_chunksize(total, worker_count)
    config_snapshot = export_runtime_config()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
        initializer=_init_extraction_worker,
        initargs=(config_snapshot,),
    ) as executor:
        for done, (idx, vec, label) in enumerate(
            executor.map(_extract_one_sample, tasks, chunksize=chunksize),
            start=1,
        ):
            vectors[idx] = vec
            labels[idx] = label
            if progress_every > 0 and (done % progress_every == 0 or done == total):
                print(f"  [{done}/{total}] completed")

    return np.stack(vectors), labels


def _feature_cache_key(method: str, n_bins: int, max_points: int) -> str:
    """Build a stable cache key that changes when feature geometry changes."""
    signature = {
        "audio": {
            "sample_rate": AudioConfig.SAMPLE_RATE,
            "n_mfcc": AudioConfig.N_MFCC,
            "include_delta": AudioConfig.INCLUDE_DELTA,
            "include_delta2": AudioConfig.INCLUDE_DELTA2,
        },
        "feature": {
            "include_f0": FeatureConfig.INCLUDE_F0,
            "include_jitter_shimmer": FeatureConfig.INCLUDE_JITTER_SHIMMER,
            "include_formants": FeatureConfig.INCLUDE_FORMANTS,
            "include_spectral_flux": FeatureConfig.INCLUDE_SPECTRAL_FLUX,
            "n_formants": FeatureConfig.N_FORMANTS,
        },
        "topology": {
            "complex": TopologyConfig.COMPLEX,
            "point_cloud_normalize": PointCloudConfig.NORMALIZE,
            "point_cloud_normalization_method": PointCloudConfig.NORMALIZATION_METHOD,
            "point_cloud_projection": PointCloudConfig.PROJECTION,
            "point_cloud_projection_dim": PointCloudConfig.PROJECTION_DIM,
            "point_cloud_projection_random_state": PointCloudConfig.PROJECTION_RANDOM_STATE,
            "max_homology_dim": TopologyConfig.MAX_HOMOLOGY_DIM,
            "distance_metric": TopologyConfig.DISTANCE_METRIC,
            "cubical_filtration": TopologyConfig.CUBICAL_FILTRATION,
            "knn_k": TopologyConfig.KNN_K,
            "knn_graph_mode": TopologyConfig.KNN_GRAPH_MODE,
            "max_edge_length": TopologyConfig.MAX_EDGE_LENGTH,
            "coeff": TopologyConfig.COEFF,
        },
        "vectorization": {
            "method": method,
            "n_bins": n_bins,
            "sigma": VectorizationConfig.PI_SIGMA,
            "landscape_n_layers": VectorizationConfig.LANDSCAPE_N_LAYERS,
            "landscape_n_bins": VectorizationConfig.LANDSCAPE_N_BINS,
            "homology_weights": VectorizationConfig.HOMOLOGY_WEIGHTS,
        },
        "point_cloud": {
            "max_points": max_points,
        },
        "spectrogram": {
            "kind": SpectrogramConfig.KIND,
            "n_mels": SpectrogramConfig.N_MELS,
            "power": SpectrogramConfig.POWER,
            "fmin": SpectrogramConfig.FMIN,
            "fmax": SpectrogramConfig.FMAX,
            "log_scale": SpectrogramConfig.LOG_SCALE,
            "compression": SpectrogramConfig.COMPRESSION,
            "smoothing": SpectrogramConfig.SMOOTHING,
            "smoothing_sigma": SpectrogramConfig.SMOOTHING_SIGMA,
            "smoothing_axis": SpectrogramConfig.SMOOTHING_AXIS,
            "band_mask_mode": SpectrogramConfig.BAND_MASK_MODE,
            "band_split_low": SpectrogramConfig.BAND_SPLIT_LOW,
            "band_split_high": SpectrogramConfig.BAND_SPLIT_HIGH,
            "band_mask_fill": SpectrogramConfig.BAND_MASK_FILL,
            "temporal_field_mode": SpectrogramConfig.TEMPORAL_FIELD_MODE,
            "temporal_field_sigma": SpectrogramConfig.TEMPORAL_FIELD_SIGMA,
            "energy_weighting_mode": SpectrogramConfig.ENERGY_WEIGHTING_MODE,
            "energy_weighting_gamma": SpectrogramConfig.ENERGY_WEIGHTING_GAMMA,
            "energy_gate_percentile": SpectrogramConfig.ENERGY_GATE_PERCENTILE,
            "energy_gate_fill": SpectrogramConfig.ENERGY_GATE_FILL,
            "normalize": SpectrogramConfig.NORMALIZE,
            "normalization_method": SpectrogramConfig.NORMALIZATION_METHOD,
            "max_frames": SpectrogramConfig.MAX_FRAMES,
        },
        "morse_smale": {
            "implementation": MorseSmaleConfig.IMPLEMENTATION,
            "graph_max_neighbors": MorseSmaleConfig.GRAPH_MAX_NEIGHBORS,
            "graph_relaxed": MorseSmaleConfig.GRAPH_RELAXED,
            "normalization": MorseSmaleConfig.NORMALIZATION,
            "simplification": MorseSmaleConfig.SIMPLIFICATION,
            "neighborhood_size": MorseSmaleConfig.NEIGHBORHOOD_SIZE,
            "top_k_basins": MorseSmaleConfig.TOP_K_BASINS,
            "include_extrema_values": MorseSmaleConfig.INCLUDE_EXTREMA_VALUES,
            "top_k_extrema": MorseSmaleConfig.TOP_K_EXTREMA,
        },
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:12]
    return f"{method}_{digest}"


def _subsample_samples(
    samples: list[tuple[Path, int]],
    max_samples: int | None,
    random_state: int,
) -> list[tuple[Path, int]]:
    """Return a reproducible subsample that preserves class balance when possible."""
    if max_samples is None or max_samples >= len(samples):
        return samples
    if max_samples <= 0:
        return []

    labels = np.array([label for _, label in samples], dtype=int)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) <= 1 or max_samples < len(unique_labels):
        return samples[:max_samples]

    rng = np.random.default_rng(random_state)
    class_indices = {
        label: rng.permutation(np.flatnonzero(labels == label))
        for label in unique_labels
    }

    target_counts = {
        label: max(1, int(np.floor(max_samples * count / len(samples))))
        for label, count in zip(unique_labels, counts)
    }

    assigned = sum(target_counts.values())
    if assigned > max_samples:
        for label in sorted(target_counts, key=target_counts.get, reverse=True):
            if assigned == max_samples:
                break
            if target_counts[label] > 1:
                target_counts[label] -= 1
                assigned -= 1
    elif assigned < max_samples:
        remainders = sorted(
            (
                (max_samples * count / len(samples)) - target_counts[label],
                label,
            )
            for label, count in zip(unique_labels, counts)
        )
        while assigned < max_samples:
            for _, label in reversed(remainders):
                if assigned == max_samples:
                    break
                if target_counts[label] < len(class_indices[label]):
                    target_counts[label] += 1
                    assigned += 1

    selected = np.concatenate(
        [class_indices[label][: target_counts[label]] for label in unique_labels]
    )
    rng.shuffle(selected)
    return [samples[idx] for idx in selected.tolist()]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        print(f"Loading config from {args.config}")
        load_config_from_yaml(str(args.config))

    method = args.method or VectorizationConfig.METHOD
    model = args.model or ClassifierConfig.MODEL
    n_bins = args.n_bins or VectorizationConfig.PI_N_BINS
    max_points = args.max_points or 300
    train_workers = args.train_workers if args.train_workers is not None else args.num_workers
    eval_workers = args.eval_workers if args.eval_workers is not None else args.num_workers

    cache_dir = args.cache_dir or (args.out_dir / "feature_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using feature cache at {cache_dir}")

    max_train_samples = args.max_train_samples
    if max_train_samples is None:
        max_train_samples = args.max_samples

    # ------------------------------------------------------------------ #
    # Mode B: train/eval                                                   #
    # ------------------------------------------------------------------ #
    if args.train_protocol is not None or args.load_model is not None:
        if args.eval_protocol is None or args.eval_audio_dir is None:
            raise ValueError(
                "train/eval mode requires --eval-protocol and --eval-audio-dir"
            )
        if args.load_model is None and (args.train_protocol is None or args.train_audio_dir is None):
            raise ValueError(
                "--train-protocol requires --train-audio-dir unless --load-model is provided"
            )

        train_cache_dir = args.train_cache_dir or (cache_dir / "train")
        eval_cache_dir = args.eval_cache_dir or (cache_dir / "eval")

        if args.load_model is None:
            print(f"Loading train samples from {args.train_protocol}...")
            train_samples = list(load_asvspoof_manifest(args.train_protocol, args.train_audio_dir))
            train_samples = _subsample_samples(
                train_samples, max_train_samples, ClassifierConfig.RANDOM_STATE
            )

            print(f"Extracting/loading features for {len(train_samples)} train samples...")
            X_train, y_train = _extract_split(
                train_samples,
                train_cache_dir,
                method,
                n_bins,
                max_points,
                num_workers=train_workers,
                progress_every=args.progress_every,
            )
        else:
            print(f"Loading saved model from {args.load_model}; skipping train extraction.")
            X_train, y_train = None, None

        print(f"Loading eval samples from {args.eval_protocol}...")
        eval_samples = list(load_asvspoof_manifest(args.eval_protocol, args.eval_audio_dir))
        eval_samples = _subsample_samples(
            eval_samples, args.max_eval_samples, ClassifierConfig.RANDOM_STATE
        )

        print(f"Extracting/loading features for {len(eval_samples)} eval samples...")
        X_eval, y_eval = _extract_split(
            eval_samples,
            eval_cache_dir,
            method,
            n_bins,
            max_points,
            num_workers=eval_workers,
            progress_every=args.progress_every,
        )

        if args.load_model is None:
            print("Training classifier...")
            clf = Classifier(
                model=model,
                svm_kernel=ClassifierConfig.SVM_KERNEL,
                svm_c=ClassifierConfig.SVM_C,
                scale_features=ClassifierConfig.SCALE_FEATURES,
                random_state=ClassifierConfig.RANDOM_STATE,
            )
            clf.fit(X_train, y_train)
            clf.save(args.out_dir / "model.pkl")
            print(f"Model saved to {args.out_dir / 'model.pkl'}")
        else:
            clf = Classifier.load(args.load_model)

        print("Evaluating on eval set...")
        eval_results = clf.evaluate(X_eval, y_eval)
        print(f"Eval AUC: {eval_results['auc']:.4f}")
        print(f"Eval EER: {eval_results['eer']:.4f}")
        print(eval_results["report"])

        metrics = {
            "auc": float(eval_results["auc"]),
            "eer": float(eval_results["eer"]),
            "n_train": None if y_train is None else len(y_train),
            "n_eval": len(y_eval),
            "n_bonafide_eval": int(np.sum(y_eval == 0)),
            "n_spoof_eval": int(np.sum(y_eval == 1)),
            "config": {
                "method": method,
                "model": model,
                "n_bins": n_bins,
                "max_points": max_points,
                "max_train_samples": max_train_samples,
                "max_eval_samples": args.max_eval_samples,
                "train_cache_dir": str(train_cache_dir),
                "eval_cache_dir": str(eval_cache_dir),
                "loaded_model": None if args.load_model is None else str(args.load_model),
            },
        }
        with open(args.out_dir / "eval_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(args.out_dir / "eval_report.txt", "w") as f:
            f.write(eval_results["report"])

        print(f"Results saved to {args.out_dir}")
        return

    # ------------------------------------------------------------------ #
    # Mode A: CV-only                                                      #
    # ------------------------------------------------------------------ #
    if args.protocol is None or args.audio_dir is None:
        raise ValueError("Provide either --protocol + --audio-dir (CV mode) or "
                         "--train-protocol + --train-audio-dir + --eval-protocol + --eval-audio-dir")

    print(f"Loading samples from {args.protocol}...")
    samples = list(load_asvspoof_manifest(args.protocol, args.audio_dir))
    samples = _subsample_samples(samples, args.max_samples, ClassifierConfig.RANDOM_STATE)

    print(f"Extracting/loading features for {len(samples)} samples...")
    X, y = _extract_split(
        samples,
        cache_dir,
        method,
        n_bins,
        max_points,
        num_workers=args.num_workers,
        progress_every=args.progress_every,
    )

    print("Running cross-validation...")
    clf = Classifier(
        model=model,
        svm_kernel=ClassifierConfig.SVM_KERNEL,
        svm_c=ClassifierConfig.SVM_C,
        scale_features=ClassifierConfig.SCALE_FEATURES,
        random_state=ClassifierConfig.RANDOM_STATE,
    )
    cv_results = clf.cross_validate(X, y, n_folds=ClassifierConfig.CV_FOLDS)
    print(f"CV accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"CV AUC:      {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}")
    print(f"CV EER:      {cv_results['eer_mean']:.3f} ± {cv_results['eer_std']:.3f}")

    with open(args.out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
