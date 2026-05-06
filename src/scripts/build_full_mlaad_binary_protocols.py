"""Build balanced binary MLAAD protocols from full MLAAD fake + M-AILABS real."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
_ENGLISH_LOCALES = ("en_US", "en_UK")
_GERMAN_LOCALES = ("de_DE",)


@dataclass(frozen=True)
class Entry:
    label: str
    language: str
    source_family: str
    source_locale: str
    path: Path


def _is_valid_audio_file(path: Path) -> bool:
    """Ignore hidden AppleDouble sidecars and non-audio files."""
    return (
        path.is_file()
        and path.suffix.lower() in _AUDIO_EXTENSIONS
        and not path.name.startswith("._")
        and not any(part.startswith("._") for part in path.parts)
    )


def _sanitize(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.")
    return text or "sample"


def _collect_full_mlaad_fake(mlaad_root: Path) -> dict[str, list[Entry]]:
    by_language: dict[str, list[Entry]] = defaultdict(list)
    fake_root = mlaad_root / "fake"
    for language_dir in sorted(fake_root.iterdir()):
        if not language_dir.is_dir():
            continue
        language = language_dir.name
        for path in sorted(p for p in language_dir.rglob("*") if _is_valid_audio_file(p)):
            by_language[language].append(
                Entry(
                    label="spoof",
                    language=language,
                    source_family="MLAAD",
                    source_locale=language,
                    path=path,
                )
            )
    return by_language


def _collect_mailabs_real(mailabs_root: Path) -> dict[str, list[Entry]]:
    mapping = {
        "en": _ENGLISH_LOCALES,
        "de": _GERMAN_LOCALES,
    }
    by_language: dict[str, list[Entry]] = defaultdict(list)
    for language, locales in mapping.items():
        for locale in locales:
            locale_root = mailabs_root / locale / locale
            if not locale_root.exists():
                continue
            for path in sorted(p for p in locale_root.rglob("*") if _is_valid_audio_file(p)):
                by_language[language].append(
                    Entry(
                        label="bonafide",
                        language=language,
                        source_family="M-AILABS",
                        source_locale=locale,
                        path=path,
                    )
                )
    return by_language


def _sample_real_entries(
    pool: list[Entry],
    target_count: int,
    *,
    seed: int,
) -> list[Entry]:
    if len(pool) < target_count:
        raise ValueError(f"Need {target_count} real samples but only found {len(pool)}")
    rng = random.Random(seed)
    items = list(pool)
    rng.shuffle(items)
    return items[:target_count]


def _allocate_counts(size: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    train_ratio, dev_ratio, test_ratio = ratios
    raw = [size * train_ratio, size * dev_ratio, size * test_ratio]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - counts[idx], raw[idx]),
        reverse=True,
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _split_entries(
    entries: list[Entry],
    *,
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, list[Entry]]:
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[Entry]] = defaultdict(list)
    for entry in entries:
        grouped[(entry.label, entry.language)].append(entry)

    splits = {"train": [], "dev": [], "test": []}
    for group_entries in grouped.values():
        items = list(group_entries)
        rng.shuffle(items)
        n_train, n_dev, n_test = _allocate_counts(len(items), ratios)
        splits["train"].extend(items[:n_train])
        splits["dev"].extend(items[n_train:n_train + n_dev])
        splits["test"].extend(items[n_train + n_dev:n_train + n_dev + n_test])

    for split_entries in splits.values():
        rng.shuffle(split_entries)
    return splits


def _materialize_dataset(
    entries: list[Entry],
    *,
    dataset_tag: str,
    out_audio_root: Path,
    out_protocol_root: Path,
    seed: int,
) -> dict[str, Path]:
    materialized_root = out_audio_root / f"{dataset_tag}_materialized"
    audio_dir = materialized_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir = out_protocol_root / f"{dataset_tag}_splits"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    metadata_rows = []
    counters = Counter()
    name_map: dict[Path, str] = {}

    for entry in entries:
        counters[entry.label] += 1
        rel_text = "__".join(entry.path.relative_to(entry.path.anchor).parts) if entry.path.is_absolute() else "__".join(entry.path.parts)
        stem = _sanitize(rel_text)
        prefix = "b" if entry.label == "bonafide" else "s"
        link_name = f"{prefix}_{counters[entry.label]:06d}_{stem}{entry.path.suffix.lower()}"
        link_path = audio_dir / link_name
        if link_path.exists() or link_path.is_symlink():
            if link_path.resolve() != entry.path.resolve():
                link_path.unlink()
                link_path.symlink_to(entry.path)
        else:
            link_path.symlink_to(entry.path)
        name_map[entry.path] = link_name
        rows.append((entry, link_name))
        metadata_rows.append(
            {
                "linked_name": link_name,
                "label": entry.label,
                "language": entry.language,
                "source_family": entry.source_family,
                "source_locale": entry.source_locale,
                "source_path": str(entry.path),
            }
        )

    protocol_all = materialized_root / "protocol.txt"
    protocol_all.write_text(
        "\n".join(f"{dataset_tag} {link_name} {entry.language} {entry.label}" for entry, link_name in rows) + "\n",
        encoding="utf-8",
    )
    (materialized_root / "summary.json").write_text(
        json.dumps(
            {
                "count": len(entries),
                "labels": dict(sorted(Counter(entry.label for entry in entries).items())),
                "languages": dict(sorted(Counter(entry.language for entry in entries).items())),
                "source_families": dict(sorted(Counter(entry.source_family for entry in entries).items())),
                "source_locales": dict(sorted(Counter(entry.source_locale for entry in entries).items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with (materialized_root / "metadata.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["linked_name", "label", "language", "source_family", "source_locale", "source_path"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    splits = _split_entries(entries, seed=seed, ratios=(0.7, 0.15, 0.15))
    split_summary: dict[str, dict[str, object]] = {}
    written: dict[str, Path] = {
        "audio_dir": audio_dir,
        "protocol_all": protocol_all,
    }
    for split_name, split_entries in splits.items():
        lines = []
        for entry in split_entries:
            lines.append(f"{dataset_tag} {name_map[entry.path]} {entry.language} {entry.label}")
        path = protocol_dir / f"{dataset_tag}_{split_name}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written[split_name] = path
        split_summary[split_name] = {
            "count": len(split_entries),
            "labels": dict(sorted(Counter(entry.label for entry in split_entries).items())),
            "languages": dict(sorted(Counter(entry.language for entry in split_entries).items())),
            "source_families": dict(sorted(Counter(entry.source_family for entry in split_entries).items())),
            "source_locales": dict(sorted(Counter(entry.source_locale for entry in split_entries).items())),
        }

    trainplusdev = protocol_dir / f"{dataset_tag}_trainplusdev.txt"
    train_lines = written["train"].read_text(encoding="utf-8").rstrip("\n")
    dev_lines = written["dev"].read_text(encoding="utf-8").rstrip("\n")
    trainplusdev.write_text(
        "\n".join(line for line in [train_lines, dev_lines] if line) + "\n",
        encoding="utf-8",
    )
    written["trainplusdev"] = trainplusdev

    (protocol_dir / f"{dataset_tag}_summary.json").write_text(
        json.dumps(split_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["summary"] = protocol_dir / f"{dataset_tag}_summary.json"
    return written


def build_dataset(
    *,
    dataset_tag: str,
    languages: tuple[str, ...],
    mlaad_fake: dict[str, list[Entry]],
    mailabs_real: dict[str, list[Entry]],
    out_audio_root: Path,
    out_protocol_root: Path,
    seed: int,
) -> dict[str, Path]:
    selected: list[Entry] = []
    for offset, language in enumerate(languages):
        fake_entries = mlaad_fake.get(language, [])
        real_entries = mailabs_real.get(language, [])
        if not fake_entries:
            raise ValueError(f"No MLAAD fake entries found for language={language}")
        if not real_entries:
            raise ValueError(f"No M-AILABS real entries found for language={language}")
        target = len(fake_entries)
        selected.extend(fake_entries)
        selected.extend(_sample_real_entries(real_entries, target, seed=seed + offset))
    return _materialize_dataset(
        selected,
        dataset_tag=dataset_tag,
        out_audio_root=out_audio_root,
        out_protocol_root=out_protocol_root,
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlaad-root", required=True)
    parser.add_argument("--mailabs-root", required=True)
    parser.add_argument("--protocol-root", required=True)
    parser.add_argument("--runtime-dataset-root", required=True)
    parser.add_argument("--tag-prefix", default="mlaad_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-german", action="store_true")
    args = parser.parse_args()

    mlaad_root = Path(args.mlaad_root).expanduser().resolve()
    mailabs_root = Path(args.mailabs_root).expanduser().resolve()
    protocol_root = Path(args.protocol_root).expanduser().resolve()
    runtime_dataset_root = Path(args.runtime_dataset_root).expanduser().resolve()

    mlaad_fake = _collect_full_mlaad_fake(mlaad_root)
    mailabs_real = _collect_mailabs_real(mailabs_root)

    outputs = {}
    outputs["en"] = build_dataset(
        dataset_tag=f"{args.tag_prefix}_en",
        languages=("en",),
        mlaad_fake=mlaad_fake,
        mailabs_real=mailabs_real,
        out_audio_root=runtime_dataset_root,
        out_protocol_root=protocol_root,
        seed=args.seed,
    )
    outputs["ende"] = build_dataset(
        dataset_tag=f"{args.tag_prefix}_ende",
        languages=("en", "de"),
        mlaad_fake=mlaad_fake,
        mailabs_real=mailabs_real,
        out_audio_root=runtime_dataset_root,
        out_protocol_root=protocol_root,
        seed=args.seed,
    )
    if not args.skip_german:
        outputs["de"] = build_dataset(
            dataset_tag=f"{args.tag_prefix}_de",
            languages=("de",),
            mlaad_fake=mlaad_fake,
            mailabs_real=mailabs_real,
            out_audio_root=runtime_dataset_root,
            out_protocol_root=protocol_root,
            seed=args.seed,
        )

    summary = {
        key: {name: str(path) for name, path in value.items()}
        for key, value in outputs.items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
