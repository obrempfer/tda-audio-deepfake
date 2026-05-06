"""Build mixed ASV2019 + full MLAAD-English train assets for cross-dataset runs."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tda_deepfake.utils import load_asvspoof_manifest


@dataclass(frozen=True)
class Entry:
    label: str
    source_family: str
    path: Path


def _sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "sample"


def _collect_entries(protocol_file: Path, audio_dir: Path, source_family: str) -> list[Entry]:
    entries: list[Entry] = []
    for audio_path, label_int in load_asvspoof_manifest(protocol_file, audio_dir):
        label = "bonafide" if label_int == 0 else "spoof"
        entries.append(
            Entry(
                label=label,
                source_family=source_family,
                path=audio_path.resolve(),
            )
        )
    if not entries:
        raise ValueError(f"No entries loaded from protocol={protocol_file}")
    return entries


def _sample_balanced(entries: list[Entry], *, target_per_label: int, seed: int) -> list[Entry]:
    grouped: dict[str, list[Entry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.label].append(entry)

    if set(grouped) != {"bonafide", "spoof"}:
        raise ValueError(f"Need both bonafide and spoof labels, saw {sorted(grouped)}")

    rng = random.Random(seed)
    selected: list[Entry] = []
    for offset, label in enumerate(("bonafide", "spoof")):
        pool = list(grouped[label])
        if len(pool) < target_per_label:
            raise ValueError(
                f"Need {target_per_label} samples for label={label}, found {len(pool)}"
            )
        rng_local = random.Random(seed + offset)
        rng_local.shuffle(pool)
        selected.extend(pool[:target_per_label])
    rng.shuffle(selected)
    return selected


def _materialize(entries: list[Entry], *, dataset_tag: str, out_root: Path) -> tuple[Path, Path, Path]:
    materialized_root = out_root / f"{dataset_tag}_materialized"
    audio_dir = materialized_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    rows: list[str] = []
    metadata: list[dict[str, str]] = []

    for entry in entries:
        counters[(entry.source_family, entry.label)] += 1
        prefix = "a" if entry.source_family == "ASV2019" else "m"
        label_prefix = "b" if entry.label == "bonafide" else "s"
        stem = _sanitize("__".join(entry.path.parts[-4:]))
        link_name = (
            f"{prefix}_{label_prefix}_{counters[(entry.source_family, entry.label)]:05d}_{stem}"
            f"{entry.path.suffix.lower()}"
        )
        link_path = audio_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            if link_path.resolve() != entry.path:
                link_path.unlink()
                link_path.symlink_to(entry.path)
        else:
            link_path.symlink_to(entry.path)
        rows.append(f"{dataset_tag} {link_name} {entry.source_family} {entry.label}")
        metadata.append(
            {
                "linked_name": link_name,
                "label": entry.label,
                "source_family": entry.source_family,
                "source_path": str(entry.path),
            }
        )

    protocol_path = materialized_root / f"{dataset_tag}.txt"
    protocol_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    summary_path = materialized_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "count": len(entries),
                "labels": dict(sorted(Counter(entry.label for entry in entries).items())),
                "source_families": dict(sorted(Counter(entry.source_family for entry in entries).items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metadata_path = materialized_root / "metadata.tsv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["linked_name", "label", "source_family", "source_path"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(metadata)
    return protocol_path, audio_dir, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asv-train-protocol", required=True)
    parser.add_argument("--asv-train-audio-dir", required=True)
    parser.add_argument("--mlaad-train-protocol", required=True)
    parser.add_argument("--mlaad-train-audio-dir", required=True)
    parser.add_argument("--runtime-dataset-root", required=True)
    parser.add_argument("--dataset-tag", default="mixed_source_full_mlaad_en_20260505")
    parser.add_argument("--asv-per-label", type=int, default=None)
    parser.add_argument("--mlaad-per-label", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asv_entries = _collect_entries(
        Path(args.asv_train_protocol).expanduser().resolve(),
        Path(args.asv_train_audio_dir).expanduser().resolve(),
        "ASV2019",
    )
    mlaad_entries = _collect_entries(
        Path(args.mlaad_train_protocol).expanduser().resolve(),
        Path(args.mlaad_train_audio_dir).expanduser().resolve(),
        "MLAAD",
    )

    label_counts = Counter(entry.label for entry in asv_entries)
    if label_counts.get("bonafide", 0) != label_counts.get("spoof", 0):
        raise ValueError(f"Expected balanced ASV source, got {dict(label_counts)}")
    asv_per_label = args.asv_per_label if args.asv_per_label is not None else label_counts["bonafide"]
    mlaad_label_counts = Counter(entry.label for entry in mlaad_entries)
    mlaad_default = min(mlaad_label_counts.get("bonafide", 0), mlaad_label_counts.get("spoof", 0))
    mlaad_per_label = args.mlaad_per_label if args.mlaad_per_label is not None else min(asv_per_label, mlaad_default)

    selected_asv = _sample_balanced(
        asv_entries,
        target_per_label=asv_per_label,
        seed=args.seed,
    )
    selected_mlaad = _sample_balanced(
        mlaad_entries,
        target_per_label=mlaad_per_label,
        seed=args.seed + 100,
    )
    mixed_entries = list(selected_asv) + list(selected_mlaad)
    random.Random(args.seed).shuffle(mixed_entries)

    protocol_path, audio_dir, summary_path = _materialize(
        mixed_entries,
        dataset_tag=args.dataset_tag,
        out_root=Path(args.runtime_dataset_root).expanduser().resolve(),
    )

    print(
        json.dumps(
            {
                "protocol": str(protocol_path),
                "audio_dir": str(audio_dir),
                "summary": str(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
