from pathlib import Path

from tda_deepfake.utils.protocol_split import (
    load_protocol_entries,
    make_stratified_protocol_splits,
    summarize_protocol_entries,
    write_protocol_splits,
)


def test_load_protocol_entries_filters_allowed_partitions(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    protocol.write_text(
        "\n".join(
            [
                "LA_0001 LA_E_0000001 alaw ita_tx A01 bonafide notrim progress",
                "LA_0002 LA_E_0000002 alaw ita_tx A01 spoof notrim eval",
                "LA_0003 LA_E_0000003 alaw ita_tx A02 spoof notrim hidden",
            ]
        )
        + "\n"
    )

    entries = load_protocol_entries(protocol, allowed_partitions={"progress", "eval"})

    assert len(entries) == 2
    assert {entry.partition for entry in entries} == {"progress", "eval"}


def test_make_stratified_protocol_splits_preserves_counts_and_labels(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    lines = []
    for attack in ("A01", "A02"):
        for index in range(12):
            label = "bonafide" if index % 2 == 0 else "spoof"
            lines.append(
                f"LA_{attack}_{index:04d} LA_E_{attack}_{index:07d} alaw ita_tx {attack} {label} notrim eval"
            )
    protocol.write_text("\n".join(lines) + "\n")

    entries = load_protocol_entries(protocol)
    splits = make_stratified_protocol_splits(entries, seed=7)

    assert sum(len(items) for items in splits.values()) == len(entries)
    for split_name in ("train", "dev", "test"):
        summary = summarize_protocol_entries(splits[split_name])
        assert summary["count"] > 0
        assert set(summary["labels"]) == {"bonafide", "spoof"}

    written = write_protocol_splits(splits, out_dir=tmp_path / "derived", prefix="internal")
    assert written["train"] == Path(tmp_path / "derived" / "internal_train.txt")
    assert written["summary"].exists()
