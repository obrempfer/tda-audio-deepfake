"""Helpers for building reproducible internal protocol splits."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
import re

_LABEL_TOKENS = {"bonafide", "spoof"}
_ATTACK_TOKEN = re.compile(r"^A\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class ProtocolEntry:
    """One labeled protocol row plus lightweight metadata used for splitting."""

    raw_line: str
    label: str
    attack: str
    partition: str | None


def load_protocol_entries(
    protocol_file: str | Path,
    *,
    allowed_partitions: set[str] | None = None,
) -> list[ProtocolEntry]:
    """Load protocol rows and extract coarse metadata for stratified splitting."""
    protocol_file = Path(protocol_file)
    entries: list[ProtocolEntry] = []

    with protocol_file.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            label = _extract_label(parts)
            if label is None:
                raise ValueError(
                    f"Could not find bonafide/spoof label in {protocol_file}:{line_no}: {line}"
                )

            partition = parts[-1].lower() if parts else None
            if allowed_partitions and partition not in allowed_partitions:
                continue

            entries.append(
                ProtocolEntry(
                    raw_line=line,
                    label=label,
                    attack=_extract_attack(parts),
                    partition=partition,
                )
            )

    if not entries:
        raise ValueError(f"No protocol entries loaded from {protocol_file}")

    return entries


def make_stratified_protocol_splits(
    entries: list[ProtocolEntry],
    *,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    group_by_attack: bool = True,
) -> dict[str, list[ProtocolEntry]]:
    """Split rows into train/dev/test while preserving label and attack balance."""
    ratios = {"train": train_ratio, "dev": dev_ratio, "test": test_ratio}
    _validate_ratios(ratios)

    rng = random.Random(seed)
    grouped: dict[str, list[ProtocolEntry]] = defaultdict(list)
    for entry in entries:
        key = entry.label if not group_by_attack else f"{entry.label}|{entry.attack}"
        grouped[key].append(entry)

    splits = {"train": [], "dev": [], "test": []}
    split_names = ["train", "dev", "test"]
    ratio_values = [ratios[name] for name in split_names]

    for group_entries in grouped.values():
        items = list(group_entries)
        rng.shuffle(items)
        counts = _allocate_counts(len(items), ratio_values)

        start = 0
        for split_name, count in zip(split_names, counts):
            end = start + count
            splits[split_name].extend(items[start:end])
            start = end

    for split_entries in splits.values():
        rng.shuffle(split_entries)

    return splits


def summarize_protocol_entries(entries: list[ProtocolEntry]) -> dict[str, object]:
    """Return compact split statistics for later inspection."""
    labels = Counter(entry.label for entry in entries)
    attacks = Counter(entry.attack for entry in entries)
    partitions = Counter(entry.partition for entry in entries if entry.partition is not None)
    return {
        "count": len(entries),
        "labels": dict(sorted(labels.items())),
        "attacks": dict(sorted(attacks.items())),
        "partitions": dict(sorted(partitions.items())),
    }


def write_protocol_splits(
    splits: dict[str, list[ProtocolEntry]],
    *,
    out_dir: str | Path,
    prefix: str,
) -> dict[str, Path]:
    """Write protocol split files plus a JSON summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = {}
    summary: dict[str, object] = {}

    for split_name, entries in splits.items():
        path = out_dir / f"{prefix}_{split_name}.txt"
        path.write_text("\n".join(entry.raw_line for entry in entries) + "\n", encoding="utf-8")
        written_paths[split_name] = path
        summary[split_name] = summarize_protocol_entries(entries)

    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written_paths["summary"] = summary_path
    return written_paths


def _extract_label(parts: list[str]) -> str | None:
    for token in reversed(parts):
        normalized = token.strip().lower()
        if normalized in _LABEL_TOKENS:
            return normalized
    return None


def _extract_attack(parts: list[str]) -> str:
    for token in parts:
        if _ATTACK_TOKEN.match(token):
            return token.upper()
    return "-"


def _validate_ratios(ratios: dict[str, float]) -> None:
    if any(value < 0.0 for value in ratios.values()):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={total})")
    if ratios["train"] <= 0.0 or ratios["dev"] <= 0.0 or ratios["test"] <= 0.0:
        raise ValueError(f"Train/dev/test ratios must all be positive, got {ratios}")


def _allocate_counts(size: int, ratios: list[float]) -> list[int]:
    raw = [size * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remainder = size - sum(counts)

    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw[idx] - counts[idx], ratios[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1

    positive_splits = [idx for idx, ratio in enumerate(ratios) if ratio > 0.0]
    if size >= len(positive_splits):
        zero_splits = [idx for idx in positive_splits if counts[idx] == 0]
        for zero_idx in zero_splits:
            donor_idx = max(
                (idx for idx in positive_splits if counts[idx] > 1),
                key=lambda idx: counts[idx],
                default=None,
            )
            if donor_idx is None:
                break
            counts[donor_idx] -= 1
            counts[zero_idx] += 1

    return counts
