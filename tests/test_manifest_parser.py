"""Unit tests for ASVspoof protocol parsing across dataset versions."""

from pathlib import Path

from tda_deepfake.utils.audio import load_asvspoof_manifest


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_manifest_parses_asvspoof2019_style_line(tmp_path: Path):
    protocol = tmp_path / "train_protocol.txt"
    audio_dir = tmp_path / "audio"
    _touch(audio_dir / "LA_T_1234567.flac")
    protocol.write_text("LA_0001 LA_T_1234567 - A01 bonafide\n")

    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    assert len(samples) == 1
    path, label = samples[0]
    assert path.name == "LA_T_1234567.flac"
    assert label == 0


def test_manifest_parses_asvspoof2021_cm_trial_metadata_line(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    audio_dir = tmp_path / "audio"
    _touch(audio_dir / "LA_E_9332881.flac")
    protocol.write_text("LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval\n")

    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    assert len(samples) == 1
    path, label = samples[0]
    assert path.name == "LA_E_9332881.flac"
    assert label == 1


def test_manifest_resolves_suffixed_utterance_tokens(tmp_path: Path):
    protocol = tmp_path / "trial_metadata.txt"
    audio_dir = tmp_path / "audio"
    _touch(audio_dir / "LA_E_5013670.flac")
    protocol.write_text(
        "LA_0007-alaw-ita_tx LA_E_5013670-alaw-ita_tx alaw ita_tx bonafide nontarget notrim eval\n"
    )

    samples = list(load_asvspoof_manifest(protocol, audio_dir))
    assert len(samples) == 1
    path, label = samples[0]
    assert path.name == "LA_E_5013670.flac"
    assert label == 0
