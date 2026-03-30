"""CLI entry point: run the full TDA detection pipeline.

Usage:
    python -m scripts.run_pipeline --help
    python -m scripts.run_pipeline --protocol data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.train.trn.txt \
        --audio-dir data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac \
        --out-dir data/results/run_01

Options:
    --protocol      Path to ASVspoof 2019 LA protocol file.
    --audio-dir     Directory containing .flac audio files.
    --out-dir       Output directory for results and ablation reports.
    --max-samples   Maximum number of samples to process (default: all).
    --method        Vectorization method: persistence_image, landscape, statistics (default: persistence_image).
    --model         Classifier: svm or logistic (default: svm).
    --ablation      Run dimensional ablation on flagged samples (flag).
"""

import argparse
import json
import numpy as np
from pathlib import Path

from tda_deepfake.utils import load_audio, load_asvspoof_manifest
from tda_deepfake.features import extract_features, build_point_cloud
from tda_deepfake.topology import compute_persistence, vectorize_diagrams
from tda_deepfake.classification import Classifier
from tda_deepfake.ablation import AblationAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TDA Audio Deepfake Detection Pipeline")
    parser.add_argument("--protocol", type=Path, required=True, help="ASVspoof protocol file")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory of .flac files")
    parser.add_argument("--out-dir", type=Path, default=Path("data/results/default"), help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on number of samples")
    parser.add_argument("--method", default="persistence_image",
                        choices=["persistence_image", "landscape", "statistics"])
    parser.add_argument("--model", default="svm", choices=["svm", "logistic"])
    parser.add_argument("--ablation", action="store_true", help="Run ablation on flagged samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading samples from {args.protocol}...")
    samples = list(load_asvspoof_manifest(args.protocol, args.audio_dir))
    if args.max_samples:
        samples = samples[: args.max_samples]

    print(f"Extracting features for {len(samples)} samples...")
    X_list, y_list = [], []
    for audio_path, label in samples:
        audio = load_audio(audio_path)
        features = extract_features(audio)
        point_cloud = build_point_cloud(features)
        diagrams = compute_persistence(point_cloud)
        vec = vectorize_diagrams(diagrams, method=args.method)
        X_list.append(vec)
        y_list.append(label)

    X = np.stack(X_list)
    y = np.array(y_list)

    print("Running cross-validation...")
    clf = Classifier(model=args.model)
    cv_results = clf.cross_validate(X, y)
    print(f"CV accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"CV AUC:      {cv_results['auc_mean']:.3f} ± {cv_results['auc_std']:.3f}")

    # Save CV results
    with open(args.out_dir / "cv_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    print(f"Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
