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
import numpy as np
from pathlib import Path

from tda_deepfake.utils import load_audio, load_asvspoof_manifest
from tda_deepfake.features import extract_features, build_point_cloud, build_mel_spectrogram
from tda_deepfake.topology import compute_persistence, vectorize_diagrams
from tda_deepfake.classification import Classifier
from tda_deepfake.config import (
    AudioConfig,
    ClassifierConfig,
    FeatureConfig,
    PointCloudConfig,
    SpectrogramConfig,
    TopologyConfig,
    VectorizationConfig,
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
                        help="Cap on number of samples (useful for quick smoke tests)")
    parser.add_argument("--method", default=None,
                        choices=["persistence_image", "landscape", "statistics"],
                        help="Vectorization method")
    parser.add_argument("--model", default=None, choices=["svm", "logistic"],
                        help="Classifier type")
    parser.add_argument("--n-bins", type=int, default=None,
                        help="Persistence image grid resolution (n_bins x n_bins)")
    parser.add_argument("--max-points", type=int, default=None,
                        help="Max point cloud size per utterance (subsampled uniformly)")
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional YAML config file to override defaults")
    parser.add_argument("--ablation", action="store_true",
                        help="Run dimensional ablation on flagged samples (not yet wired)")
    return parser.parse_args()


def _extract_split(
    samples: list,
    cache_dir: Path,
    method: str,
    n_bins: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (or load from cache) feature vectors for a list of samples.

    Args:
        samples: List of (audio_path, label) pairs.
        cache_dir: Directory for .npy cache files.
        method: Vectorization method passed to vectorize_diagrams().
        n_bins: Grid resolution for persistence images.
        max_points: Maximum point cloud size (subsampled if exceeded).

    Returns:
        (X, y) numpy arrays.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    X_list, y_list = [], []

    for i, (audio_path, label) in enumerate(samples):
        if i % 100 == 0:
            print(f"  [{i}/{len(samples)}] {audio_path.name}")

        cache_file = cache_dir / f"{audio_path.stem}_{_feature_cache_key(method, n_bins, max_points)}.npy"
        if cache_file.exists():
            vec = np.load(cache_file)
        else:
            audio = load_audio(audio_path, sample_rate=AudioConfig.SAMPLE_RATE)
            if TopologyConfig.COMPLEX == "cubical":
                topological_object = build_mel_spectrogram(
                    audio,
                    sample_rate=AudioConfig.SAMPLE_RATE,
                    n_mels=SpectrogramConfig.N_MELS,
                    power=SpectrogramConfig.POWER,
                    fmin=SpectrogramConfig.FMIN,
                    fmax=SpectrogramConfig.FMAX,
                    log_scale=SpectrogramConfig.LOG_SCALE,
                    normalize=SpectrogramConfig.NORMALIZE,
                    normalization_method=SpectrogramConfig.NORMALIZATION_METHOD,
                    max_frames=SpectrogramConfig.MAX_FRAMES,
                )
            else:
                features = extract_features(
                    audio,
                    sample_rate=AudioConfig.SAMPLE_RATE,
                    include_delta=AudioConfig.INCLUDE_DELTA,
                    include_delta2=AudioConfig.INCLUDE_DELTA2,
                    include_f0=FeatureConfig.INCLUDE_F0,
                    include_jitter_shimmer=FeatureConfig.INCLUDE_JITTER_SHIMMER,
                    include_formants=FeatureConfig.INCLUDE_FORMANTS,
                    include_spectral_flux=FeatureConfig.INCLUDE_SPECTRAL_FLUX,
                )
                topological_object = build_point_cloud(
                    features,
                    max_points=max_points,
                    normalize=PointCloudConfig.NORMALIZE,
                    normalization_method=PointCloudConfig.NORMALIZATION_METHOD,
                    projection=PointCloudConfig.PROJECTION,
                    projection_dim=PointCloudConfig.PROJECTION_DIM,
                    projection_random_state=PointCloudConfig.PROJECTION_RANDOM_STATE,
                )

            diagrams = compute_persistence(
                topological_object,
                complex_type=TopologyConfig.COMPLEX,
                max_dim=TopologyConfig.MAX_HOMOLOGY_DIM,
                metric=TopologyConfig.DISTANCE_METRIC,
                cubical_filtration=TopologyConfig.CUBICAL_FILTRATION,
                max_edge_length=TopologyConfig.MAX_EDGE_LENGTH,
                coeff=TopologyConfig.COEFF,
            )
            vec = vectorize_diagrams(
                diagrams,
                method=method,
                n_bins=n_bins,
                sigma=VectorizationConfig.PI_SIGMA,
            )
            np.save(cache_file, vec)

        X_list.append(vec)
        y_list.append(label)

    return np.stack(X_list), np.array(y_list)


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
            "max_edge_length": TopologyConfig.MAX_EDGE_LENGTH,
            "coeff": TopologyConfig.COEFF,
        },
        "vectorization": {
            "method": method,
            "n_bins": n_bins,
            "sigma": VectorizationConfig.PI_SIGMA,
            "landscape_n_layers": VectorizationConfig.LANDSCAPE_N_LAYERS,
            "landscape_n_bins": VectorizationConfig.LANDSCAPE_N_BINS,
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
            "normalize": SpectrogramConfig.NORMALIZE,
            "normalization_method": SpectrogramConfig.NORMALIZATION_METHOD,
            "max_frames": SpectrogramConfig.MAX_FRAMES,
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

    labels = np.array([label for _, label in samples], dtype=int)
    unique_labels, counts = np.unique(labels, return_counts=True)

    if len(unique_labels) <= 1:
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

    cache_dir = args.out_dir / "feature_cache"

    # ------------------------------------------------------------------ #
    # Mode B: train/eval                                                   #
    # ------------------------------------------------------------------ #
    if args.train_protocol is not None:
        if args.train_audio_dir is None or args.eval_protocol is None or args.eval_audio_dir is None:
            raise ValueError(
                "--train-protocol requires --train-audio-dir, --eval-protocol, and --eval-audio-dir"
            )

        print(f"Loading train samples from {args.train_protocol}...")
        train_samples = list(load_asvspoof_manifest(args.train_protocol, args.train_audio_dir))
        train_samples = _subsample_samples(
            train_samples, args.max_samples, ClassifierConfig.RANDOM_STATE
        )

        print(f"Extracting features for {len(train_samples)} train samples...")
        X_train, y_train = _extract_split(
            train_samples, cache_dir / "train", method, n_bins, max_points
        )

        print(f"Loading eval samples from {args.eval_protocol}...")
        eval_samples = list(load_asvspoof_manifest(args.eval_protocol, args.eval_audio_dir))

        print(f"Extracting features for {len(eval_samples)} eval samples...")
        X_eval, y_eval = _extract_split(
            eval_samples, cache_dir / "eval", method, n_bins, max_points
        )

        print("Training classifier...")
        clf = Classifier(
            model=model,
            svm_kernel=ClassifierConfig.SVM_KERNEL,
            svm_c=ClassifierConfig.SVM_C,
            random_state=ClassifierConfig.RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        clf.save(args.out_dir / "model.pkl")
        print(f"Model saved to {args.out_dir / 'model.pkl'}")

        print("Evaluating on eval set...")
        eval_results = clf.evaluate(X_eval, y_eval)
        print(f"Eval AUC: {eval_results['auc']:.4f}")
        print(eval_results["report"])

        metrics = {
            "auc": float(eval_results["auc"]),
            "n_train": len(y_train),
            "n_eval": len(y_eval),
            "n_bonafide_eval": int(np.sum(y_eval == 0)),
            "n_spoof_eval": int(np.sum(y_eval == 1)),
            "config": {
                "method": method,
                "model": model,
                "n_bins": n_bins,
                "max_points": max_points,
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

    print(f"Extracting features for {len(samples)} samples...")
    X, y = _extract_split(samples, cache_dir, method, n_bins, max_points)

    print("Running cross-validation...")
    clf = Classifier(
        model=model,
        svm_kernel=ClassifierConfig.SVM_KERNEL,
        svm_c=ClassifierConfig.SVM_C,
        random_state=ClassifierConfig.RANDOM_STATE,
    )
    cv_results = clf.cross_validate(X, y, n_folds=ClassifierConfig.CV_FOLDS)
    print(f"CV accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"CV AUC:      {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}")

    with open(args.out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
