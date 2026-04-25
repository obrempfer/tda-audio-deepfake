"""Train topology-only linear / flat-MLP / staged-MLP experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    bootstrap_str = str(bootstrap_path)
    if bootstrap_str not in sys.path:
        sys.path.insert(0, bootstrap_str)

from src.scripts.run_pipeline import _extract_split, _subsample_samples
from tda_deepfake.config import (
    ClassifierConfig,
    VectorizationConfig,
    apply_runtime_config,
    export_runtime_config,
    load_config_from_yaml,
)
from tda_deepfake.neural import (
    FeatureLayout,
    StageSpec,
    TopologyLinearBaseline,
    TopologyMLP,
    evaluate_block_ablations,
    stack_feature_blocks,
)
from tda_deepfake.utils import load_asvspoof_manifest


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    protocol: Path
    audio_dir: Path
    max_samples: int | None


@dataclass(frozen=True)
class FeatureBlockSpec:
    name: str
    config_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topology-only neural experiments")
    parser.add_argument("--train-protocol", type=Path, required=True)
    parser.add_argument("--train-audio-dir", type=Path, required=True)
    parser.add_argument("--val-protocol", type=Path, required=True)
    parser.add_argument("--val-audio-dir", type=Path, required=True)
    parser.add_argument("--eval-protocol", type=Path, required=True)
    parser.add_argument("--eval-audio-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-eval",
        action="append",
        default=[],
        help="Extra eval dataset in the form name=protocol_path::audio_dir[::max_samples]",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/tmp") / "topology_nn_cache")
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-val-samples", type=int, default=5000)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument(
        "--model-kind",
        choices=["linear", "flat_mlp", "staged_mlp", "all"],
        default="all",
    )
    parser.add_argument("--core-config", type=Path, default=REPO_ROOT / "configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml")
    parser.add_argument("--aux-a-config", type=Path, default=REPO_ROOT / "configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml")
    parser.add_argument("--aux-b-config", type=Path, default=REPO_ROOT / "configs/experiments/cubical_mel_best_field_svm.yaml")
    parser.add_argument("--skip-aux-b", action="store_true", help="Omit the broad full-field auxiliary block")
    parser.add_argument("--linear-c", type=float, default=1.0)
    parser.add_argument("--hidden-dims", default="128,64", help="Comma-separated hidden-layer sizes for the MLP")
    parser.add_argument("--feature-dropout", type=float, default=0.10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--monitor", choices=["auc", "eer", "accuracy"], default="eer")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--flat-epochs", type=int, default=40)
    parser.add_argument("--flat-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stage-epochs", default="25,20,15")
    parser.add_argument("--stage-learning-rates", default="1e-3,3e-4,1e-4")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    block_specs = build_block_specs(args)
    datasets = build_datasets(args)
    feature_payload = materialize_feature_payload(
        block_specs=block_specs,
        datasets=datasets,
        cache_root=args.cache_root,
        workers=args.workers,
        progress_every=args.progress_every,
    )
    layout = feature_payload["layout"]

    selected_models = (
        ["linear", "flat_mlp", "staged_mlp"]
        if args.model_kind == "all"
        else [args.model_kind]
    )
    hidden_dims = tuple(int(part) for part in args.hidden_dims.split(",") if part.strip())
    stage_epochs = parse_int_list(args.stage_epochs)
    stage_lrs = parse_float_list(args.stage_learning_rates)

    train_X = feature_payload["datasets"]["train"]["X"]
    train_y = feature_payload["datasets"]["train"]["y"]
    val_X = feature_payload["datasets"]["val"]["X"]
    val_y = feature_payload["datasets"]["val"]["y"]

    summary: dict[str, object] = {
        "block_layout": layout.to_dict(),
        "blocks": {
            spec.name: {"config_path": str(spec.config_path)}
            for spec in block_specs
        },
        "datasets": {
            name: {
                "protocol": str(spec.protocol),
                "audio_dir": str(spec.audio_dir),
                "max_samples": spec.max_samples,
                "n_samples": int(len(feature_payload["datasets"][name]["y"])),
            }
            for name, spec in datasets.items()
        },
        "models": {},
    }

    for model_name in selected_models:
        model_dir = args.out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        if model_name == "linear":
            model = TopologyLinearBaseline(
                c=args.linear_c,
                random_state=args.random_state,
            )
            model.fit(train_X, train_y)
            training_meta = {
                "model": "logistic_regression",
                "c": args.linear_c,
            }
        elif model_name == "flat_mlp":
            model = TopologyMLP(
                hidden_layer_sizes=hidden_dims,
                alpha=args.weight_decay,
                feature_dropout=args.feature_dropout,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            stages = [
                StageSpec(
                    name="flat_all_blocks",
                    active_blocks=tuple(layout.block_names),
                    max_epochs=args.flat_epochs,
                    learning_rate=args.flat_learning_rate,
                )
            ]
            model.fit(
                train_X,
                train_y,
                val_X,
                val_y,
                layout=layout,
                stages=stages,
                monitor=args.monitor,
                patience=args.patience,
            )
            training_meta = {
                "model": "flat_mlp",
                "hidden_dims": list(hidden_dims),
                "feature_dropout": args.feature_dropout,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "monitor": args.monitor,
                "patience": args.patience,
                "stages": [asdict(stage) for stage in stages],
                "history": model.training_history_,
                "best_validation_metrics": model.best_validation_metrics_,
                "fitted_stages": model.fitted_stages_,
            }
        else:
            model = TopologyMLP(
                hidden_layer_sizes=hidden_dims,
                alpha=args.weight_decay,
                feature_dropout=args.feature_dropout,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            stages = build_staged_specs(layout, stage_epochs=stage_epochs, stage_lrs=stage_lrs)
            model.fit(
                train_X,
                train_y,
                val_X,
                val_y,
                layout=layout,
                stages=stages,
                monitor=args.monitor,
                patience=args.patience,
            )
            training_meta = {
                "model": "staged_mlp",
                "hidden_dims": list(hidden_dims),
                "feature_dropout": args.feature_dropout,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "monitor": args.monitor,
                "patience": args.patience,
                "stages": [asdict(stage) for stage in stages],
                "history": model.training_history_,
                "best_validation_metrics": model.best_validation_metrics_,
                "fitted_stages": model.fitted_stages_,
            }

        metrics_by_dataset = {}
        ablations_by_dataset = {}
        for dataset_name, payload in feature_payload["datasets"].items():
            if dataset_name == "train":
                continue
            metrics = model.evaluate(payload["X"], payload["y"])
            ablations = evaluate_block_ablations(model, payload["X"], payload["y"], layout)
            metrics_by_dataset[dataset_name] = metrics
            ablations_by_dataset[dataset_name] = ablations

        model.save(model_dir / "model.joblib")
        results = {
            "training": training_meta,
            "metrics": metrics_by_dataset,
            "ablations": ablations_by_dataset,
        }
        (model_dir / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["models"][model_name] = results

    (args.out_dir / "comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote comparison to {args.out_dir / 'comparison.json'}")


def build_block_specs(args: argparse.Namespace) -> list[FeatureBlockSpec]:
    specs = [
        FeatureBlockSpec(name="core_lowband_h1", config_path=args.core_config),
        FeatureBlockSpec(name="aux_lowband_h0", config_path=args.aux_a_config),
    ]
    if not args.skip_aux_b:
        specs.append(FeatureBlockSpec(name="aux_fullfield_h0_h1", config_path=args.aux_b_config))
    return specs


def build_datasets(args: argparse.Namespace) -> dict[str, DatasetSpec]:
    datasets = {
        "train": DatasetSpec(
            name="train",
            protocol=args.train_protocol,
            audio_dir=args.train_audio_dir,
            max_samples=args.max_train_samples,
        ),
        "val": DatasetSpec(
            name="val",
            protocol=args.val_protocol,
            audio_dir=args.val_audio_dir,
            max_samples=args.max_val_samples,
        ),
        "eval": DatasetSpec(
            name="eval",
            protocol=args.eval_protocol,
            audio_dir=args.eval_audio_dir,
            max_samples=args.max_eval_samples,
        ),
    }
    for raw in args.extra_eval:
        name, spec = raw.split("=", 1)
        parts = spec.split("::")
        if len(parts) not in {2, 3}:
            raise ValueError(
                f"Invalid --extra-eval {raw!r}; expected name=protocol::audio_dir[::max_samples]"
            )
        protocol = Path(parts[0])
        audio_dir = Path(parts[1])
        max_samples = None if len(parts) == 2 or parts[2] == "" else int(parts[2])
        datasets[name] = DatasetSpec(
            name=name,
            protocol=protocol,
            audio_dir=audio_dir,
            max_samples=max_samples,
        )
    return datasets


def materialize_feature_payload(
    *,
    block_specs: list[FeatureBlockSpec],
    datasets: dict[str, DatasetSpec],
    cache_root: Path,
    workers: int,
    progress_every: int,
) -> dict[str, object]:
    default_snapshot = export_runtime_config()
    per_dataset: dict[str, dict[str, object]] = {}
    layout: FeatureLayout | None = None

    for dataset_name, dataset in datasets.items():
        samples = list(load_asvspoof_manifest(dataset.protocol, dataset.audio_dir))
        samples = _subsample_samples(samples, dataset.max_samples, ClassifierConfig.RANDOM_STATE)
        block_arrays: dict[str, np.ndarray] = {}
        label_ref: np.ndarray | None = None

        for block_spec in block_specs:
            apply_runtime_config(default_snapshot)
            load_config_from_yaml(str(block_spec.config_path))
            method = VectorizationConfig.METHOD
            n_bins = VectorizationConfig.PI_N_BINS
            max_points = 300
            cache_dir = cache_root / block_spec.name / dataset_name
            X_block, y_block = _extract_split(
                samples,
                cache_dir=cache_dir,
                method=method,
                n_bins=n_bins,
                max_points=max_points,
                num_workers=workers,
                progress_every=progress_every,
            )
            if label_ref is None:
                label_ref = y_block
            else:
                if not np.array_equal(label_ref, y_block):
                    raise ValueError(f"Label mismatch while building {dataset_name} block {block_spec.name}")
            block_arrays[block_spec.name] = np.asarray(X_block, dtype=np.float64)

        X_full, dataset_layout = stack_feature_blocks(block_arrays)
        if layout is None:
            layout = dataset_layout
        else:
            if layout.to_dict() != dataset_layout.to_dict():
                raise ValueError(f"Feature layout mismatch for dataset {dataset_name}")

        per_dataset[dataset_name] = {
            "X": X_full,
            "y": np.asarray(label_ref, dtype=int),
            "block_dims": {
                name: int(matrix.shape[1])
                for name, matrix in block_arrays.items()
            },
        }

    if layout is None:
        raise ValueError("No datasets materialized")

    return {
        "layout": layout,
        "datasets": per_dataset,
    }


def build_staged_specs(
    layout: FeatureLayout,
    *,
    stage_epochs: list[int],
    stage_lrs: list[float],
) -> list[StageSpec]:
    block_names = layout.block_names
    if len(block_names) < 2:
        raise ValueError("Need at least core and one auxiliary block for staged training")

    active_sets = [
        (block_names[0],),
        tuple(block_names[:2]),
    ]
    if len(block_names) >= 3:
        active_sets.append(tuple(block_names))

    while len(stage_epochs) < len(active_sets):
        stage_epochs.append(stage_epochs[-1])
    while len(stage_lrs) < len(active_sets):
        stage_lrs.append(stage_lrs[-1])

    stage_names = ["stage1_core", "stage2_core_plus_aux_a", "stage3_core_plus_aux_a_plus_aux_b"]
    specs = []
    for idx, active_blocks in enumerate(active_sets):
        specs.append(
            StageSpec(
                name=stage_names[idx],
                active_blocks=active_blocks,
                max_epochs=stage_epochs[idx],
                learning_rate=stage_lrs[idx],
            )
        )
    return specs


def parse_int_list(raw: str) -> list[int]:
    out = [int(part) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one integer")
    return out


def parse_float_list(raw: str) -> list[float]:
    out = [float(part) for part in raw.split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one float")
    return out


if __name__ == "__main__":
    main()
