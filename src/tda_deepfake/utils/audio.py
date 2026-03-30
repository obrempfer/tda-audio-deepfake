"""Audio I/O and ASVspoof dataset loading utilities."""

import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Iterator

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from ..config import AudioConfig


def load_audio(
    path: str | Path,
    sample_rate: int = AudioConfig.SAMPLE_RATE,
    mono: bool = True,
) -> npt.NDArray[np.float32]:
    """Load an audio file and resample to the target sample rate.

    Args:
        path: Path to a WAV (or other librosa-supported) audio file.
        sample_rate: Target sample rate in Hz.
        mono: If True, mix down to mono.

    Returns:
        1-D float32 numpy array of audio samples.

    Raises:
        ImportError: If librosa is not installed.
        FileNotFoundError: If the audio file does not exist.
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio loading. pip install librosa")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, _ = librosa.load(path, sr=sample_rate, mono=mono)
    return audio


def load_asvspoof_manifest(
    protocol_file: str | Path,
    audio_dir: str | Path,
) -> Iterator[tuple[Path, int]]:
    """Iterate over (audio_path, label) pairs from an ASVspoof 2019 protocol file.

    ASVspoof 2019 LA protocol format (space-separated):
        SPEAKER_ID  UTTERANCE_ID  -  ATTACK_TYPE  LABEL
    where LABEL is 'bonafide' (real=0) or 'spoof' (fake=1).

    Args:
        protocol_file: Path to the ASVspoof protocol .txt file
            (e.g., ASVspoof2019.LA.cm.train.trn.txt).
        audio_dir: Directory containing the .flac audio files.

    Yields:
        Tuples of (audio_path, label) where label is 0 (real) or 1 (fake).
    """
    protocol_file = Path(protocol_file)
    audio_dir = Path(audio_dir)

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            utterance_id = parts[1]
            label_str = parts[4]
            label = 0 if label_str == "bonafide" else 1
            audio_path = audio_dir / f"{utterance_id}.flac"
            yield audio_path, label
