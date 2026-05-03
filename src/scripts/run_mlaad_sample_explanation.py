"""Run a bounded MLAAD English sample-level explanation case study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

def _default_repo_root() -> Path:
    env_root = os.environ.get("TDA_REPO_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "src").exists() and (cwd / "configs").exists():
        return cwd

    resolved = Path(__file__).resolve()
    for parent in resolved.parents:
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent

    raise RuntimeError("Could not infer repository root; pass --repo-root explicitly.")


REPO_ROOT = _default_repo_root()
SRC_ROOT = REPO_ROOT / "src"
for bootstrap_path in (REPO_ROOT, SRC_ROOT):
    bootstrap_str = str(bootstrap_path)
    if bootstrap_str not in sys.path:
        sys.path.insert(0, bootstrap_str)

from src.scripts.run_pipeline import _compute_feature_vector, _extract_split
from tda_deepfake.classification import Classifier
from tda_deepfake.config import ClassifierConfig, VectorizationConfig, load_config_from_yaml
from tda_deepfake.utils import load_asvspoof_manifest, load_protocol_entries


@dataclass(frozen=True)
class SampleEntry:
    audio_path: Path
    label: int
    utt_id: str
    raw_line: str

    @property
    def label_name(self) -> str:
        return "spoof" if self.label == 1 else "bonafide"


@dataclass(frozen=True)
class VariantSpec:
    key: str
    branch: str
    display_name: str
    config_path: Path
    model_path: Path


def parse_args() -> argparse.Namespace:
    user = os.environ.get("USER", "user")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--run-tag", default="mlaad_diag_en_20260501")
    parser.add_argument(
        "--protocol-root",
        type=Path,
        default=REPO_ROOT / "data" / "protocols" / "mlaad_tiny",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(f"/tmp/{user}/tda_deepfake_runtime/sample_explanations"),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def load_test_entries(protocol_path: Path, audio_dir: Path) -> list[SampleEntry]:
    metas = load_protocol_entries(protocol_path)
    resolved = list(load_asvspoof_manifest(protocol_path, audio_dir))
    if len(metas) != len(resolved):
        raise ValueError("Protocol rows and resolved samples differ")

    out = []
    for meta, (audio_path, label) in zip(metas, resolved):
        parts = meta.raw_line.split()
        utt_id = parts[1] if len(parts) > 1 else audio_path.stem
        out.append(SampleEntry(audio_path=audio_path, label=label, utt_id=utt_id, raw_line=meta.raw_line))
    return out


def true_label_support(prob_fake: float, label: int) -> float:
    return prob_fake if label == 1 else (1.0 - prob_fake)


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


def score_single_sample(sample: SampleEntry, variant: VariantSpec, cache_dir: Path) -> dict[str, float | int | str]:
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
    pred = 1 if prob_fake >= 0.5 else 0
    return {
        "variant": variant.display_name,
        "prob_fake": prob_fake,
        "pred_label": pred,
        "pred_label_name": "spoof" if pred == 1 else "bonafide",
        "true_label_support": true_label_support(prob_fake, sample.label),
    }


def pick_case(
    entries: list[SampleEntry],
    ms_scores: np.ndarray,
    cubical_scores: np.ndarray,
    *,
    want_label: int,
    mode: str,
) -> SampleEntry:
    ranked: list[tuple[float, SampleEntry]] = []
    for entry, ms_score, cub_score in zip(entries, ms_scores, cubical_scores):
        if entry.label != want_label:
            continue
        ms_pred = 1 if ms_score >= 0.5 else 0
        cub_pred = 1 if cub_score >= 0.5 else 0
        ms_support = true_label_support(float(ms_score), entry.label)
        cub_support = true_label_support(float(cub_score), entry.label)

        if mode == "ms_easy":
            if ms_pred == entry.label:
                ranked.append((ms_support, entry))
        elif mode == "ms_beats_cubical":
            if ms_pred == entry.label and cub_pred != entry.label:
                ranked.append((ms_support - cub_support, entry))
        elif mode == "shared_failure":
            if ms_pred != entry.label and cub_pred != entry.label:
                ranked.append((max(float(ms_score), float(cub_score)), entry))
        elif mode == "cubical_failure":
            if cub_pred != entry.label and ms_pred == entry.label:
                ranked.append((ms_support - cub_support, entry))
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

    if not ranked:
        return pick_fallback(entries, ms_scores, want_label)

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def pick_fallback(entries: list[SampleEntry], ms_scores: np.ndarray, want_label: int) -> SampleEntry:
    ranked = []
    for entry, ms_score in zip(entries, ms_scores):
        if entry.label != want_label:
            continue
        ranked.append((true_label_support(float(ms_score), entry.label), entry))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def english_variants(
    repo_root: Path,
    results_root: Path,
    run_tag: str,
    cache_root: Path,
) -> tuple[dict[str, VariantSpec], dict[str, VariantSpec]]:
    generated_root = Path(f"/tmp/{os.environ.get('USER', 'user')}/tda_deepfake_runtime/generated_mlaad_diagnostic_configs/{run_tag}")

    cubical = {
        "full_reference": VariantSpec(
            "full_reference",
            "cubical",
            "Cubical full",
            repo_root / "configs/experiments/cubical_mel_best_field_svm.yaml",
            results_root / f"{run_tag}_cubical_full_reference/model.pkl",
        ),
        "keep_low_gate10": VariantSpec(
            "keep_low_gate10",
            "cubical",
            "Cubical keep_low gate10",
            repo_root / "configs/experiments/ablation/cubical_best_band_keep_low.yaml",
            results_root / f"{run_tag}_cubical_keep_low_gate10/model.pkl",
        ),
        "keep_low_gate12": VariantSpec(
            "keep_low_gate12",
            "cubical",
            "Cubical keep_low gate12",
            repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_gate12.yaml",
            results_root / f"{run_tag}_cubical_keep_low_gate12/model.pkl",
        ),
        "drop_low": VariantSpec(
            "drop_low",
            "cubical",
            "Cubical drop_low",
            repo_root / "configs/experiments/ablation/cubical_best_band_drop_low.yaml",
            results_root / f"{run_tag}_cubical_drop_low/model.pkl",
        ),
        "h1_only": VariantSpec(
            "h1_only",
            "cubical",
            "Cubical H1 only",
            repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_h1_only.yaml",
            results_root / f"{run_tag}_cubical_h1_only/model.pkl",
        ),
        "h0_only": VariantSpec(
            "h0_only",
            "cubical",
            "Cubical H0 only",
            repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_h0_only.yaml",
            results_root / f"{run_tag}_cubical_h0_only/model.pkl",
        ),
        "gate_off": VariantSpec(
            "gate_off",
            "cubical",
            "Cubical gate_off",
            repo_root / "configs/experiments/ablation/cubical_best_band_keep_low_gate_off.yaml",
            results_root / f"{run_tag}_cubical_gate_off/model.pkl",
        ),
    }

    morse = {}
    for key, display_name in [
        ("full_reference", "Morse full"),
        ("keep_low_gate10", "Morse keep_low gate10"),
        ("keep_low_gate12", "Morse keep_low gate12"),
        ("drop_low", "Morse drop_low"),
        ("gate_off", "Morse gate_off"),
        ("counts_entropy", "Morse counts+entropy"),
        ("basin_fractions", "Morse basin fractions"),
        ("merge_sequence", "Morse merge sequence"),
        ("extrema_values", "Morse extrema values"),
    ]:
        morse[key] = VariantSpec(
            key,
            "morse",
            display_name,
            generated_root / f"morse_{key}.yaml",
            results_root / f"{run_tag}_morse_{key}/model.pkl",
        )

    return cubical, morse


def ensure_models(variants: dict[str, VariantSpec]) -> None:
    missing = [str(spec.model_path) for spec in variants.values() if not spec.model_path.exists()]
    if missing:
        raise FileNotFoundError("Missing variant models:\n" + "\n".join(missing))


def render_markdown(
    cases: list[tuple[str, SampleEntry]],
    cubical_scores: dict[str, dict[str, dict[str, float | int | str]]],
    morse_scores: dict[str, dict[str, dict[str, float | int | str]]],
) -> str:
    lines = [
        "# MLAAD English Sample-Level Explanation",
        "",
        "Scores report `prob_fake` and `support` (`P(fake)` for spoof, `1-P(fake)` for bonafide).",
        "",
    ]

    cubical_order = ["full_reference", "keep_low_gate10", "keep_low_gate12", "drop_low", "h1_only", "h0_only", "gate_off"]
    morse_order = ["full_reference", "keep_low_gate10", "keep_low_gate12", "drop_low", "gate_off", "counts_entropy", "basin_fractions", "merge_sequence", "extrema_values"]

    for case_key, sample in cases:
        lines.extend(
            [
                f"## {case_key}",
                "",
                f"- File: `{sample.audio_path.name}`",
                f"- Label: `{sample.label_name}`",
                "",
                "| Cubical variant | prob_fake | support | pred |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for key in cubical_order:
            score = cubical_scores[case_key][key]
            lines.append(
                f"| {score['variant']} | {float(score['prob_fake']):.3f} | {float(score['true_label_support']):.3f} | {score['pred_label_name']} |"
            )

        lines.extend(
            [
                "",
                "| Morse variant | prob_fake | support | pred |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for key in morse_order:
            score = morse_scores[case_key][key]
            lines.append(
                f"| {score['variant']} | {float(score['prob_fake']):.3f} | {float(score['true_label_support']):.3f} | {score['pred_label_name']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    results_root = (
        args.results_root.expanduser().resolve()
        if args.results_root is not None
        else (repo_root / "data/results").resolve()
    )
    run_tag = args.run_tag
    split_root = args.protocol_root / f"{run_tag}_splits"
    materialized_root = args.protocol_root / f"{run_tag}_materialized"
    protocol_path = split_root / f"{run_tag}_test.txt"
    audio_dir = materialized_root / "audio"

    cubical_variants, morse_variants = english_variants(
        repo_root,
        results_root,
        run_tag,
        args.cache_root,
    )
    ensure_models(cubical_variants)
    ensure_models(morse_variants)

    entries = load_test_entries(protocol_path, audio_dir)
    cubical_anchor_scores = batch_score_entries(
        entries,
        cubical_variants["keep_low_gate12"],
        args.cache_root / "anchor_cubical",
        args.workers,
        args.progress_every,
    )
    morse_anchor_scores = batch_score_entries(
        entries,
        morse_variants["keep_low_gate10"],
        args.cache_root / "anchor_morse",
        args.workers,
        args.progress_every,
    )

    cases = [
        ("fake_ms_easy", pick_case(entries, morse_anchor_scores, cubical_anchor_scores, want_label=1, mode="ms_easy")),
        ("fake_ms_beats_cubical", pick_case(entries, morse_anchor_scores, cubical_anchor_scores, want_label=1, mode="ms_beats_cubical")),
        ("bonafide_ms_beats_cubical", pick_case(entries, morse_anchor_scores, cubical_anchor_scores, want_label=0, mode="cubical_failure")),
        ("shared_failure", pick_case(entries, morse_anchor_scores, cubical_anchor_scores, want_label=1, mode="shared_failure")),
    ]

    cubical_scores: dict[str, dict[str, dict[str, float | int | str]]] = {}
    morse_scores: dict[str, dict[str, dict[str, float | int | str]]] = {}
    for case_key, sample in cases:
        cubical_scores[case_key] = {
            key: score_single_sample(sample, variant, args.cache_root / f"{case_key}_{key}")
            for key, variant in cubical_variants.items()
        }
        morse_scores[case_key] = {
            key: score_single_sample(sample, variant, args.cache_root / f"{case_key}_{key}")
            for key, variant in morse_variants.items()
        }

    out_dir = results_root / f"{run_tag}_sample_explanations"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases.json").write_text(
        json.dumps(
            {
                "cases": [
                    {"case_key": case_key, "utt_id": sample.utt_id, "label": sample.label_name, "audio_path": str(sample.audio_path)}
                    for case_key, sample in cases
                ],
                "cubical_scores": cubical_scores,
                "morse_scores": morse_scores,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_markdown(cases, cubical_scores, morse_scores), encoding="utf-8")
    print(f"Wrote sample explanations to {out_dir}")


if __name__ == "__main__":
    main()
