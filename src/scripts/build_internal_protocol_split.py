"""Build a reproducible internal train/dev/test split from an ASVspoof protocol."""

from __future__ import annotations

import argparse

from tda_deepfake.utils.protocol_split import (
    load_protocol_entries,
    make_stratified_protocol_splits,
    summarize_protocol_entries,
    write_protocol_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Source protocol file")
    parser.add_argument("--out-dir", required=True, help="Directory for derived split files")
    parser.add_argument("--prefix", required=True, help="Filename prefix for generated splits")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Train split ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.2, help="Dev split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument(
        "--allowed-partitions",
        default="",
        help="Comma-separated partition tokens to retain, e.g. 'progress,eval'. Empty keeps all rows.",
    )
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="Stratify only by bonafide/spoof instead of label+attack.",
    )
    args = parser.parse_args()

    allowed_partitions = {
        token.strip().lower()
        for token in args.allowed_partitions.split(",")
        if token.strip()
    } or None

    entries = load_protocol_entries(args.protocol, allowed_partitions=allowed_partitions)
    splits = make_stratified_protocol_splits(
        entries,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        group_by_attack=not args.label_only,
    )
    written_paths = write_protocol_splits(splits, out_dir=args.out_dir, prefix=args.prefix)

    source_summary = summarize_protocol_entries(entries)
    print(f"Loaded {source_summary['count']} rows from {args.protocol}")
    for split_name in ("train", "dev", "test"):
        summary = summarize_protocol_entries(splits[split_name])
        print(
            f"{split_name}: {summary['count']} rows "
            f"labels={summary['labels']} attacks={len(summary['attacks'])} "
            f"path={written_paths[split_name]}"
        )
    print(f"Summary: {written_paths['summary']}")


if __name__ == "__main__":
    main()
