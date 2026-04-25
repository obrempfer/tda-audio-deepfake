"""Build a small sample-level explanation case study across recent cubical runs.

The demo is intentionally bounded. It selects a handful of representative
samples, scores them under the most relevant branch variants, and renders a
compact markdown report plus spectrogram figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    bootstrap_str = str(bootstrap_path)
    if bootstrap_str not in sys.path:
        sys.path.insert(0, bootstrap_str)

from src.scripts.run_pipeline import _compute_feature_vector, _extract_split
from tda_deepfake.classification import Classifier
from tda_deepfake.config import (
    AudioConfig,
    ClassifierConfig,
    SpectrogramConfig,
    VectorizationConfig,
    load_config_from_yaml,
)
from tda_deepfake.features import build_raw_mel_spectrogram, postprocess_mel_spectrogram
from tda_deepfake.utils import load_asvspoof_manifest, load_audio, load_protocol_entries


@dataclass(frozen=True)
class SampleEntry:
    dataset: str
    split: str
    audio_path: Path
    utt_id: str
    label: int
    attack: str
    raw_line: str

    @property
    def label_name(self) -> str:
        return "spoof" if self.label == 1 else "bonafide"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    display_name: str
    config_path: Path
    model_path: Path


@dataclass(frozen=True)
class CaseSelection:
    key: str
    display_name: str
    family_key: str
    sample: SampleEntry


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "user")

    parser = argparse.ArgumentParser(description="Run the sample-level explanation mini-demo.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root containing configs/, data/, and src/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where report, JSON, CSV, and figures will be written.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(f"/tmp/{user}/tda_sample_explanation_cache"),
        help="Shared feature-cache root for this demo.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(f"/tmp/{user}/tda_results"),
        help="Results root that holds the 2021 internal-split models.",
    )
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=Path(f"/tmp/{user}/tda_protocols/asvspoof2021_la_internal_seed42"),
        help="Directory that holds the internal 2021 LA split files.",
    )
    parser.add_argument(
        "--la2021-run-tag",
        default="la2021_internal_topology_bg16_20260424",
        help="Run tag prefix for the 2021 LA internal topology sweep.",
    )
    parser.add_argument(
        "--search-limit-2019",
        type=int,
        default=256,
        help="How many 2019 dev rows to scan when choosing the representative fake.",
    )
    parser.add_argument(
        "--search-limit-2021",
        type=int,
        default=512,
        help="How many 2021 internal-dev rows to scan when choosing the fake / bona fide / failure cases.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker processes used for bounded candidate-pool feature extraction.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print extraction progress every N completed samples during bounded scans.",
    )
    parser.add_argument(
        "--materialize-2019-drop-low",
        action="store_true",
        help="Train and save the missing 2019 holdout drop_low model if it is absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    families = build_variant_families(
        repo_root=args.repo_root,
        results_root=args.results_root,
        la2021_run_tag=args.la2021_run_tag,
    )

    if args.materialize_2019_drop_low:
        materialize_2019_drop_low_model(
            repo_root=args.repo_root,
            variant=families["holdout2019"]["drop_low"],
            cache_root=args.cache_root / "holdout2019_model_train",
            workers=args.workers,
            progress_every=args.progress_every,
        )

    require_variant_models(families)

    sample_pools = load_sample_pools(
        repo_root=args.repo_root,
        protocol_root=args.protocol_root,
        search_limit_2019=args.search_limit_2019,
        search_limit_2021=args.search_limit_2021,
    )

    selected_cases = select_cases(sample_pools, families, args.cache_root, args.workers, args.progress_every)
    case_scores = score_cases(selected_cases, families, args.cache_root)
    figure_paths = render_case_figures(selected_cases, families, figures_dir)

    report = build_report(selected_cases, case_scores, figure_paths)
    write_outputs(args.out_dir, selected_cases, case_scores, figure_paths, report)

    print(f"Wrote report to {args.out_dir / 'sample_explanation_demo.md'}")
    print(f"Wrote JSON to {args.out_dir / 'sample_explanation_demo.json'}")
    print(f"Wrote CSV to {args.out_dir / 'sample_explanation_demo_scores.csv'}")


def build_variant_families(
    *,
    repo_root: Path,
    results_root: Path,
    la2021_run_tag: str,
) -> dict[str, dict[str, VariantSpec]]:
    configs = {
        "reference": repo_root / "configs/experiments/cubical_mel_best_field_svm.yaml",
        "keep_low": repo_root / "configs/experiments/ablation/cubical_best_band_keep_low.yaml",
        "drop_low": repo_root / "configs/experiments/ablation/cubical_best_band_drop_low.yaml",
        "keep_low_h1": repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml",
        "keep_low_h0": repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml",
        "gate_off": repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml",
    }

    return {
        "holdout2019": {
            "reference": VariantSpec(
                "reference",
                "full_reference",
                configs["reference"],
                repo_root / "data/results/holdout_all_full_20260418_154849_phaseA_holdout_reference/model.pkl",
            ),
            "keep_low": VariantSpec(
                "keep_low",
                "keep_low (gate10)",
                configs["keep_low"],
                repo_root / "data/results/holdout_all_full_20260418_154849_phaseA_holdout_keep_low/model.pkl",
            ),
            "drop_low": VariantSpec(
                "drop_low",
                "drop_low",
                configs["drop_low"],
                repo_root / "data/results/sample_explanation_2019_drop_low/model.pkl",
            ),
            "keep_low_h1": VariantSpec(
                "keep_low_h1",
                "keep_low_h1",
                configs["keep_low_h1"],
                repo_root / "data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h1/model.pkl",
            ),
            "keep_low_h0": VariantSpec(
                "keep_low_h0",
                "keep_low_h0",
                configs["keep_low_h0"],
                repo_root / "data/results/holdout_all_full_20260418_154849_phaseB_holdout_keep_low_h0/model.pkl",
            ),
            "gate_off": VariantSpec(
                "gate_off",
                "keep_low gate_off",
                configs["gate_off"],
                repo_root / "data/results/cross_dataset_followups_bg16_20260424_keep_low_gate_off/model.pkl",
            ),
        },
        "la2021_internal": {
            "reference": VariantSpec(
                "reference",
                "full_reference",
                configs["reference"],
                results_root / f"{la2021_run_tag}_full_reference/model.pkl",
            ),
            "keep_low": VariantSpec(
                "keep_low",
                "keep_low (gate10)",
                configs["keep_low"],
                results_root / f"{la2021_run_tag}_keep_low/model.pkl",
            ),
            "drop_low": VariantSpec(
                "drop_low",
                "drop_low",
                configs["drop_low"],
                results_root / f"{la2021_run_tag}_drop_low/model.pkl",
            ),
            "keep_low_h1": VariantSpec(
                "keep_low_h1",
                "keep_low_h1",
                configs["keep_low_h1"],
                results_root / f"{la2021_run_tag}_keep_low_h1/model.pkl",
            ),
            "keep_low_h0": VariantSpec(
                "keep_low_h0",
                "keep_low_h0",
                configs["keep_low_h0"],
                results_root / f"{la2021_run_tag}_keep_low_h0/model.pkl",
            ),
            "gate_off": VariantSpec(
                "gate_off",
                "keep_low gate_off",
                configs["gate_off"],
                results_root / f"{la2021_run_tag}_gate_off/model.pkl",
            ),
        },
    }


def require_variant_models(families: dict[str, dict[str, VariantSpec]]) -> None:
    missing = [
        str(variant.model_path)
        for family in families.values()
        for variant in family.values()
        if not variant.model_path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required model files:\n" + "\n".join(sorted(missing)))


def materialize_2019_drop_low_model(
    *,
    repo_root: Path,
    variant: VariantSpec,
    cache_root: Path,
    workers: int,
    progress_every: int,
) -> None:
    if variant.model_path.exists():
        return

    print(f"Materializing missing 2019 drop_low model at {variant.model_path}")
    load_config_from_yaml(str(variant.config_path))

    train_protocol = repo_root / "data/raw/ASVspoof2019_LA/derived/ASVspoof2019.LA.cm.train.all_bonafide_balanced.seed42.txt"
    train_audio_dir = repo_root / "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_train/flac"
    train_samples = list(load_asvspoof_manifest(train_protocol, train_audio_dir))
    train_samples = subsample_balanced(train_samples, max_samples=1000, random_state=ClassifierConfig.RANDOM_STATE)

    X_train, y_train = _extract_split(
        train_samples,
        cache_root / "train",
        method=VectorizationConfig.METHOD,
        n_bins=VectorizationConfig.PI_N_BINS,
        max_points=300,
        num_workers=workers,
        progress_every=progress_every,
    )

    clf = Classifier(
        model=ClassifierConfig.MODEL,
        svm_kernel=ClassifierConfig.SVM_KERNEL,
        svm_c=ClassifierConfig.SVM_C,
        scale_features=ClassifierConfig.SCALE_FEATURES,
        random_state=ClassifierConfig.RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    variant.model_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(variant.model_path)
    print(f"Saved 2019 drop_low model to {variant.model_path}")


def subsample_balanced(
    samples: list[tuple[Path, int]],
    *,
    max_samples: int | None,
    random_state: int,
) -> list[tuple[Path, int]]:
    if max_samples is None or max_samples >= len(samples):
        return samples
    if max_samples <= 0:
        return []

    labels = np.array([label for _, label in samples], dtype=int)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) <= 1 or max_samples < len(unique_labels):
        return samples[:max_samples]

    rng = np.random.default_rng(random_state)
    class_indices = {label: rng.permutation(np.flatnonzero(labels == label)) for label in unique_labels}
    target_counts = {
        label: max(1, int(np.floor(max_samples * count / len(samples))))
        for label, count in zip(unique_labels, counts)
    }

    assigned = sum(target_counts.values())
    while assigned > max_samples:
        for label in sorted(target_counts, key=target_counts.get, reverse=True):
            if assigned == max_samples:
                break
            if target_counts[label] > 1:
                target_counts[label] -= 1
                assigned -= 1

    while assigned < max_samples:
        for label in sorted(target_counts, key=target_counts.get):
            if assigned == max_samples:
                break
            if target_counts[label] < len(class_indices[label]):
                target_counts[label] += 1
                assigned += 1

    selected = np.concatenate([class_indices[label][: target_counts[label]] for label in unique_labels])
    rng.shuffle(selected)
    return [samples[idx] for idx in selected.tolist()]


def load_sample_pools(
    *,
    repo_root: Path,
    protocol_root: Path,
    search_limit_2019: int,
    search_limit_2021: int,
) -> dict[str, list[SampleEntry]]:
    sample_pools = {}
    sample_pools["holdout2019"] = load_sample_entries(
        dataset="2019_LA",
        split="dev",
        protocol_path=repo_root / "data/raw/ASVspoof2019_LA/ASVspoof2019.LA.cm.dev.trl.txt",
        audio_dir=repo_root / "data/raw/ASVspoof2019_LA/ASVspoof2019_LA_dev/flac",
        limit=search_limit_2019,
    )
    sample_pools["la2021_internal"] = load_sample_entries(
        dataset="2021_LA",
        split="internal_dev",
        protocol_path=protocol_root / "asvspoof2021_la_internal_seed42_dev.txt",
        audio_dir=repo_root / "data/raw/ASVspoof2021_LA/ASVspoof2021_LA_eval/flac",
        limit=search_limit_2021,
    )
    return sample_pools


def load_sample_entries(
    *,
    dataset: str,
    split: str,
    protocol_path: Path,
    audio_dir: Path,
    limit: int | None,
) -> list[SampleEntry]:
    entries = load_protocol_entries(protocol_path)
    samples = list(load_asvspoof_manifest(protocol_path, audio_dir))

    if len(entries) != len(samples):
        raise ValueError(
            f"Protocol parsing mismatch for {protocol_path}: "
            f"{len(entries)} metadata rows vs {len(samples)} resolved samples"
        )

    if limit is not None:
        entries = entries[:limit]
        samples = samples[:limit]

    out = []
    for meta, (audio_path, label) in zip(entries, samples):
        parts = meta.raw_line.split()
        utt_id = parts[1] if len(parts) > 1 else audio_path.stem
        normalized = 1 if meta.label == "spoof" else 0
        if normalized != label:
            raise ValueError(f"Label mismatch for {utt_id} in {protocol_path}")
        out.append(
            SampleEntry(
                dataset=dataset,
                split=split,
                audio_path=audio_path,
                utt_id=utt_id,
                label=label,
                attack=meta.attack,
                raw_line=meta.raw_line,
            )
        )
    return out


def select_cases(
    sample_pools: dict[str, list[SampleEntry]],
    families: dict[str, dict[str, VariantSpec]],
    cache_root: Path,
    workers: int,
    progress_every: int,
) -> list[CaseSelection]:
    selected: list[CaseSelection] = []

    scores_2019 = batch_score_entries(
        sample_pools["holdout2019"],
        families["holdout2019"]["keep_low"],
        cache_root / "holdout2019_select_keep_low",
        workers,
        progress_every,
    )
    scores_2021 = batch_score_entries(
        sample_pools["la2021_internal"],
        families["la2021_internal"]["keep_low"],
        cache_root / "la2021_internal_select_keep_low",
        workers,
        progress_every,
    )

    selected.append(
        CaseSelection(
            key="case_2019_fake",
            display_name="2019 LA fake",
            family_key="holdout2019",
            sample=pick_highest_confidence(
                sample_pools["holdout2019"],
                scores_2019,
                want_label=1,
                predicted_correct=True,
            ),
        )
    )
    selected.append(
        CaseSelection(
            key="case_2021_fake",
            display_name="2021 LA fake",
            family_key="la2021_internal",
            sample=pick_highest_confidence(
                sample_pools["la2021_internal"],
                scores_2021,
                want_label=1,
                predicted_correct=True,
            ),
        )
    )
    selected.append(
        CaseSelection(
            key="case_2021_bonafide",
            display_name="2021 LA bona fide",
            family_key="la2021_internal",
            sample=pick_highest_confidence(
                sample_pools["la2021_internal"],
                scores_2021,
                want_label=0,
                predicted_correct=True,
            ),
        )
    )
    selected.append(
        CaseSelection(
            key="case_2021_failure",
            display_name="2021 LA failure case",
            family_key="la2021_internal",
            sample=pick_failure_case(sample_pools["la2021_internal"], scores_2021),
        )
    )
    return selected


def batch_score_entries(
    entries: list[SampleEntry],
    variant: VariantSpec,
    cache_dir: Path,
    workers: int,
    progress_every: int,
) -> np.ndarray:
    load_config_from_yaml(str(variant.config_path))
    samples = [(entry.audio_path, entry.label) for entry in entries]
    X, _ = _extract_split(
        samples,
        cache_dir,
        method=VectorizationConfig.METHOD,
        n_bins=VectorizationConfig.PI_N_BINS,
        max_points=300,
        num_workers=workers,
        progress_every=progress_every,
    )
    clf = Classifier.load(variant.model_path)
    return clf.predict_proba(X)[:, 1]


def pick_highest_confidence(
    entries: list[SampleEntry],
    scores: np.ndarray,
    *,
    want_label: int,
    predicted_correct: bool,
) -> SampleEntry:
    ranked = []
    for entry, score in zip(entries, scores):
        if entry.label != want_label:
            continue
        pred = 1 if score >= 0.5 else 0
        if predicted_correct and pred != entry.label:
            continue
        support = true_label_support(score, entry.label)
        ranked.append((support, entry))

    if not ranked:
        raise ValueError(f"No candidate found for label={want_label} predicted_correct={predicted_correct}")

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def pick_failure_case(entries: list[SampleEntry], scores: np.ndarray) -> SampleEntry:
    failures = []
    near_boundary = []
    for entry, score in zip(entries, scores):
        pred = 1 if score >= 0.5 else 0
        if pred != entry.label:
            confidence_wrong = score if entry.label == 0 else (1.0 - score)
            failures.append((confidence_wrong, entry))
        near_boundary.append((abs(score - 0.5), entry))

    if failures:
        failures.sort(key=lambda item: item[0], reverse=True)
        return failures[0][1]

    near_boundary.sort(key=lambda item: item[0])
    return near_boundary[0][1]


def score_cases(
    cases: list[CaseSelection],
    families: dict[str, dict[str, VariantSpec]],
    cache_root: Path,
) -> dict[str, dict[str, dict[str, float | int | str]]]:
    results: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for case in cases:
        family = families[case.family_key]
        results[case.key] = {}
        for variant_key in ["reference", "keep_low", "drop_low", "keep_low_h1", "keep_low_h0", "gate_off"]:
            variant = family[variant_key]
            results[case.key][variant_key] = score_single_sample(
                case.sample,
                variant,
                cache_root / f"{case.family_key}_{variant_key}",
            )
    return results


def score_single_sample(
    sample: SampleEntry,
    variant: VariantSpec,
    cache_dir: Path,
) -> dict[str, float | int | str]:
    load_config_from_yaml(str(variant.config_path))
    vec = _compute_feature_vector(
        sample.audio_path,
        cache_dir=cache_dir,
        method=VectorizationConfig.METHOD,
        n_bins=VectorizationConfig.PI_N_BINS,
        max_points=300,
    )

    clf = Classifier.load(variant.model_path)
    prob_fake = float(clf.predict_proba(vec[None, :])[0, 1])
    decision_margin = None
    if hasattr(clf.pipeline, "decision_function"):
        margin = clf.pipeline.decision_function(vec[None, :])
        decision_margin = float(np.ravel(margin)[0])

    pred_label = 1 if prob_fake >= 0.5 else 0
    return {
        "variant": variant.display_name,
        "prob_fake": prob_fake,
        "decision_margin": decision_margin,
        "pred_label": pred_label,
        "pred_label_name": "spoof" if pred_label == 1 else "bonafide",
        "true_label_support": true_label_support(prob_fake, sample.label),
    }


def render_case_figures(
    cases: list[CaseSelection],
    families: dict[str, dict[str, VariantSpec]],
    figures_dir: Path,
) -> dict[str, Path]:
    figure_paths: dict[str, Path] = {}
    figure_variants = ["reference", "keep_low", "drop_low", "gate_off"]

    for case in cases:
        raw_audio = load_audio(case.sample.audio_path, sample_rate=AudioConfig.SAMPLE_RATE)
        figure_path = figures_dir / f"{case.key}.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        axes = np.ravel(axes)

        for ax, variant_key in zip(axes, figure_variants):
            variant = families[case.family_key][variant_key]
            load_config_from_yaml(str(variant.config_path))
            raw_grid = build_raw_mel_spectrogram(
                raw_audio,
                sample_rate=AudioConfig.SAMPLE_RATE,
                n_mels=SpectrogramConfig.N_MELS,
                power=SpectrogramConfig.POWER,
                fmin=SpectrogramConfig.FMIN,
                fmax=SpectrogramConfig.FMAX,
            )
            grid = postprocess_mel_spectrogram(
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
            )
            ax.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
            ax.set_title(variant.display_name, fontsize=11)
            ax.set_xlabel("frame")
            ax.set_ylabel("mel bin")

        fig.suptitle(
            f"{case.display_name}: {case.sample.utt_id} ({case.sample.label_name}, attack={case.sample.attack})",
            fontsize=13,
        )
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)
        figure_paths[case.key] = figure_path

    return figure_paths


def build_report(
    cases: list[CaseSelection],
    case_scores: dict[str, dict[str, dict[str, float | int | str]]],
    figure_paths: dict[str, Path],
) -> str:
    lines: list[str] = []
    lines.append("# Sample-Level Explanation Mini-Demo")
    lines.append("")
    lines.append("This bounded demo scores a few representative samples under the most informative cubical branches.")
    lines.append("`keep_low` is the gated low-band branch (`gate10`), and `gate_off` is the same low-band field without the energy gate.")
    lines.append("")
    lines.append("## Score Table")
    lines.append("")
    lines.append("| Case | Dataset | Label | Attack | full_reference | keep_low | drop_low | keep_low_h1 | keep_low_h0 | gate_off |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in cases:
        scores = case_scores[case.key]
        reference = float(scores["reference"]["prob_fake"])
        row = [
            case.display_name,
            case.sample.dataset,
            case.sample.label_name,
            case.sample.attack,
        ]
        for variant_key in ["reference", "keep_low", "drop_low", "keep_low_h1", "keep_low_h0", "gate_off"]:
            prob = float(scores[variant_key]["prob_fake"])
            delta = prob - reference
            cell = f"{prob:.3f}"
            if variant_key != "reference":
                cell += f" ({delta:+.3f})"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Case Notes")
    lines.append("")
    for case in cases:
        scores = case_scores[case.key]
        figure_path = figure_paths[case.key]
        lines.append(f"### {case.display_name}")
        lines.append("")
        lines.append(
            f"- Sample: `{case.sample.utt_id}` from `{case.sample.dataset}` `{case.sample.split}` "
            f"({case.sample.label_name}, attack `{case.sample.attack}`)."
        )
        lines.extend(format_case_interpretation(case, scores))
        lines.append(f"- Figure: ![{case.display_name}](figures/{figure_path.name})")
        lines.append("")
    return "\n".join(lines) + "\n"


def format_case_interpretation(
    case: CaseSelection,
    scores: dict[str, dict[str, float | int | str]],
) -> list[str]:
    keep_support = float(scores["keep_low"]["true_label_support"])
    drop_support = float(scores["drop_low"]["true_label_support"])
    h1_support = float(scores["keep_low_h1"]["true_label_support"])
    h0_support = float(scores["keep_low_h0"]["true_label_support"])
    gate_off_support = float(scores["gate_off"]["true_label_support"])
    ref_support = float(scores["reference"]["true_label_support"])

    bullets = []
    bullets.append(
        f"- True-label support: reference `{ref_support:.3f}`, keep_low `{keep_support:.3f}`, "
        f"drop_low `{drop_support:.3f}`, H1 `{h1_support:.3f}`, H0 `{h0_support:.3f}`, gate_off `{gate_off_support:.3f}`."
    )

    band_delta = keep_support - drop_support
    if band_delta > 0.05:
        bullets.append(
            f"- Low-band emphasis helps this sample: keep_low beats drop_low by `{band_delta:+.3f}` support."
        )
    elif band_delta < -0.05:
        bullets.append(
            f"- The low-band-only branch hurts this sample relative to drop_low by `{band_delta:+.3f}` support."
        )
    else:
        bullets.append(f"- Low-band masking changes support only mildly (`{band_delta:+.3f}`).")

    homology_delta = h1_support - h0_support
    if homology_delta > 0.05:
        bullets.append(f"- H1 carries more of the useful signal than H0-only here (`{homology_delta:+.3f}`).")
    elif homology_delta < -0.05:
        bullets.append(f"- H0-only is stronger than H1 on this sample (`{homology_delta:+.3f}`).")
    else:
        bullets.append(f"- H1 and H0-only stay close on this sample (`{homology_delta:+.3f}`).")

    gate_delta = keep_support - gate_off_support
    if gate_delta > 0.05:
        bullets.append(f"- The energy gate helps on this sample (`{gate_delta:+.3f}` support over gate_off).")
    elif gate_delta < -0.05:
        bullets.append(f"- Removing the gate helps on this sample (`{gate_delta:+.3f}` for gated minus gate_off).")
    else:
        bullets.append(f"- Gating changes support only slightly (`{gate_delta:+.3f}`).")

    if case.key.endswith("failure"):
        bullets.append("- This is the intentional failure case, so at least one branch stays on the wrong side of the boundary.")

    return bullets


def true_label_support(prob_fake: float, label: int) -> float:
    return prob_fake if label == 1 else (1.0 - prob_fake)


def write_outputs(
    out_dir: Path,
    cases: list[CaseSelection],
    case_scores: dict[str, dict[str, dict[str, float | int | str]]],
    figure_paths: dict[str, Path],
    report: str,
) -> None:
    json_payload = {
        "cases": [
            {
                "case_key": case.key,
                "display_name": case.display_name,
                "family_key": case.family_key,
                "sample": {
                    **asdict(case.sample),
                    "audio_path": str(case.sample.audio_path),
                },
                "scores": case_scores[case.key],
                "figure": str(figure_paths[case.key]),
            }
            for case in cases
        ]
    }
    (out_dir / "sample_explanation_demo.json").write_text(
        json.dumps(json_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "sample_explanation_demo_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "case_key",
                "display_name",
                "dataset",
                "split",
                "utt_id",
                "label",
                "attack",
                "variant",
                "prob_fake",
                "decision_margin",
                "pred_label",
                "true_label_support",
            ]
        )
        for case in cases:
            for variant_key, metrics in case_scores[case.key].items():
                writer.writerow(
                    [
                        case.key,
                        case.display_name,
                        case.sample.dataset,
                        case.sample.split,
                        case.sample.utt_id,
                        case.sample.label_name,
                        case.sample.attack,
                        variant_key,
                        metrics["prob_fake"],
                        metrics["decision_margin"],
                        metrics["pred_label_name"],
                        metrics["true_label_support"],
                    ]
                )

    (out_dir / "sample_explanation_demo.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
