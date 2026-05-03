"""Materialize a balanced MLAAD subset with sanitized filenames and protocol."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def _collect_audio(root: Path, languages: list[str]) -> dict[str, list[Path]]:
    buckets = {"bonafide": [], "spoof": []}

    for language in languages:
        original_dir = root / "original" / language
        fake_dir = root / "fake" / language
        if original_dir.exists():
            buckets["bonafide"].extend(
                sorted(path for path in original_dir.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS)
            )
        if fake_dir.exists():
            buckets["spoof"].extend(
                sorted(path for path in fake_dir.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS)
            )
    return buckets


def _sanitize_stem(text: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_.")
    return sanitized or "sample"


def _select_balanced(
    buckets: dict[str, list[Path]],
    max_per_class: int | None,
    seed: int,
) -> dict[str, list[Path]]:
    selected = {}
    rng = np.random.default_rng(seed)

    target = min(len(buckets["bonafide"]), len(buckets["spoof"]))
    if max_per_class is not None:
        target = min(target, max_per_class)

    for label, paths in buckets.items():
        if target <= 0:
            selected[label] = []
            continue
        indices = rng.permutation(len(paths))[:target]
        selected[label] = [paths[idx] for idx in sorted(indices)]
    return selected


def _materialize_subset(
    selected: dict[str, list[Path]],
    src_root: Path,
    out_dir: Path,
) -> tuple[Path, Path, dict[str, object]]:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = out_dir / "protocol.txt"

    rows: list[str] = []
    summary_rows = []

    counters = Counter()
    for label in ("bonafide", "spoof"):
        tag = "bonafide" if label == "bonafide" else "spoof"
        for path in selected[label]:
            counters[label] += 1
            rel = path.relative_to(src_root)
            stem = _sanitize_stem("__".join(rel.parts))
            link_name = f"{label[:1]}_{counters[label]:05d}_{stem}{path.suffix.lower()}"
            link_path = audio_dir / link_name
            if not link_path.exists():
                link_path.symlink_to(path)
            rows.append(f"mlaad {link_name} {tag}")
            summary_rows.append(
                {
                    "linked_name": link_name,
                    "label": tag,
                    "source_path": str(path),
                    "language": rel.parts[1] if len(rel.parts) > 1 else "unknown",
                }
            )

    protocol_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    summary_path = out_dir / "summary.json"
    summary = {
        "count": len(rows),
        "labels": {label: len(selected[label]) for label in selected},
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return protocol_path, audio_dir, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", required=True, help="MLAAD root containing original/ and fake/")
    parser.add_argument("--out-dir", required=True, help="Output directory for symlinked subset")
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated languages to include, e.g. 'en' or 'en,de'",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=1000,
        help="Maximum number of bonafide and spoof files to materialize per class",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    languages = [token.strip() for token in args.languages.split(",") if token.strip()]

    buckets = _collect_audio(src_root, languages=languages)
    selected = _select_balanced(buckets, max_per_class=args.max_per_class, seed=args.seed)
    protocol_path, audio_dir, summary = _materialize_subset(selected, src_root=src_root, out_dir=out_dir)

    print(f"Materialized MLAAD subset at {out_dir}")
    print(f"Languages: {','.join(languages)}")
    print(f"Protocol: {protocol_path}")
    print(f"Audio dir: {audio_dir}")
    print(f"Counts: {summary['labels']}")


if __name__ == "__main__":
    main()
