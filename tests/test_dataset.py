"""Validate ASVspoof 2019 LA download is complete and parseable.

All tests are skipped if the dataset directory does not exist yet.
Run after downloading and extracting the dataset to data/raw/ASVspoof2019_LA/.
"""

import pytest
from pathlib import Path

DATA_ROOT = Path("data/raw/ASVspoof2019_LA")
DATASET_MISSING = not DATA_ROOT.exists()
skip_no_data = pytest.mark.skipif(DATASET_MISSING, reason="ASVspoof 2019 LA dataset not found")


@skip_no_data
def test_train_protocol_parseable():
    from tda_deepfake.utils.audio import load_asvspoof_manifest

    protocol = DATA_ROOT / "ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"

    assert protocol.exists(), f"Protocol file not found: {protocol}"
    assert audio_dir.exists(), f"Audio directory not found: {audio_dir}"

    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    assert len(samples) == 25380, f"Expected 25380 train samples, got {len(samples)}"

    labels = [label for _, label in samples]
    assert 0 in labels, "No bonafide (real) samples found"
    assert 1 in labels, "No spoof (fake) samples found"

    n_bonafide = labels.count(0)
    n_spoof = labels.count(1)
    assert n_bonafide == 2580, f"Expected 2580 bonafide, got {n_bonafide}"
    assert n_spoof == 22800, f"Expected 22800 spoof, got {n_spoof}"


@skip_no_data
def test_first_train_file_loads():
    from tda_deepfake.utils.audio import load_asvspoof_manifest, load_audio

    protocol = DATA_ROOT / "ASVspoof2019.LA.cm.train.trn.txt"
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
    samples = list(load_asvspoof_manifest(protocol, audio_dir))

    first_path, label = samples[0]
    assert first_path.exists(), f"Audio file not found: {first_path}"

    audio = load_audio(first_path)
    assert audio.ndim == 1, "Expected 1-D audio array"
    assert len(audio) > 0, "Audio array is empty"
    assert label in (0, 1), f"Unexpected label value: {label}"


@skip_no_data
def test_dev_protocol_parseable():
    from tda_deepfake.utils.audio import load_asvspoof_manifest

    protocol = DATA_ROOT / "ASVspoof2019.LA.cm.dev.trl.txt"
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    if not protocol.exists():
        pytest.skip("Dev protocol not found")

    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    assert len(samples) == 24844, f"Expected 24844 dev samples, got {len(samples)}"
