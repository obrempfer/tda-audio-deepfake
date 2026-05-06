"""Build train/eval feature matrices from an existing per-sample cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_pipeline import _feature_cache_key
from tda_deepfake.config import PointCloudConfig, VectorizationConfig, load_config_from_yaml
from tda_deepfake.utils import load_asvspoof_manifest


def _load_split(protocol: Path, audio_dir: Path, cache_dir: Path, cache_key: str) -> tuple[np.ndarray, np.ndarray]:
    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    vectors: list[np.ndarray] = []
    labels: list[int] = []

    missing: list[str] = []
    for audio_path, label in samples:
        cache_file = cache_dir / f"{audio_path.stem}_{cache_key}.npy"
        if not cache_file.exists():
            missing.append(str(cache_file))
            continue
        vectors.append(np.load(cache_file, allow_pickle=False))
        labels.append(int(label))

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} cached feature files under {cache_dir}. "
            f"First missing entries:\n{preview}"
        )

    return np.stack(vectors), np.asarray(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-protocol", type=Path, required=True)
    parser.add_argument("--train-audio-dir", type=Path, required=True)
    parser.add_argument("--train-cache-dir", type=Path, required=True)
    parser.add_argument("--eval-protocol", type=Path, required=True)
    parser.add_argument("--eval-audio-dir", type=Path, required=True)
    parser.add_argument("--eval-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=300)
    parser.add_argument("--n-bins", type=int, default=None)
    args = parser.parse_args()

    load_config_from_yaml(str(args.config))

    method = VectorizationConfig.METHOD
    n_bins = args.n_bins if args.n_bins is not None else VectorizationConfig.PI_N_BINS
    max_points = args.max_points
    cache_key = _feature_cache_key(method, n_bins, max_points)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _load_split(
        args.train_protocol, args.train_audio_dir, args.train_cache_dir, cache_key
    )
    X_eval, y_eval = _load_split(
        args.eval_protocol, args.eval_audio_dir, args.eval_cache_dir, cache_key
    )

    np.save(args.out_dir / "X_train.npy", X_train)
    np.save(args.out_dir / "y_train.npy", y_train)
    np.save(args.out_dir / "X_eval.npy", X_eval)
    np.save(args.out_dir / "y_eval.npy", y_eval)

    metadata = {
        "config": str(args.config),
        "method": method,
        "n_bins": n_bins,
        "max_points": max_points,
        "cache_key": cache_key,
        "train_protocol": str(args.train_protocol),
        "train_audio_dir": str(args.train_audio_dir),
        "train_cache_dir": str(args.train_cache_dir),
        "eval_protocol": str(args.eval_protocol),
        "eval_audio_dir": str(args.eval_audio_dir),
        "eval_cache_dir": str(args.eval_cache_dir),
        "n_train": int(len(y_train)),
        "n_eval": int(len(y_eval)),
        "n_features": int(X_train.shape[1]),
        "point_cloud_max_points": max_points,
        "point_cloud_normalize": bool(PointCloudConfig.NORMALIZE),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
