"""Audio I/O and ASVspoof dataset loading utilities."""

import numpy as np
import numpy.typing as npt
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from typing import Iterator

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import imageio_ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from ..config import AudioConfig

_LABEL_MAP = {
    "bonafide": 0,
    "bona-fide": 0,
    "real": 0,
    "genuine": 0,
    "spoof": 1,
    "fake": 1,
}
_AUDIO_EXTENSIONS = (".flac", ".wav")


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
    if not LIBROSA_AVAILABLE and not SOUNDFILE_AVAILABLE and not FFMPEG_AVAILABLE:
        raise ImportError("librosa, soundfile, or imageio-ffmpeg is required for audio loading")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if SOUNDFILE_AVAILABLE:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                return _read_audio_soundfile(path, sample_rate=sample_rate, mono=mono)
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(0.25 * (attempt + 1))
        try:
            with tempfile.NamedTemporaryFile(suffix=path.suffix) as tmp:
                shutil.copyfile(path, tmp.name)
                return _read_audio_soundfile(Path(tmp.name), sample_rate=sample_rate, mono=mono)
        except Exception as exc:
            last_error = exc
        if not LIBROSA_AVAILABLE:
            if FFMPEG_AVAILABLE:
                return _read_audio_ffmpeg(path, sample_rate=sample_rate, mono=mono)
            raise last_error

    if FFMPEG_AVAILABLE:
        try:
            return _read_audio_ffmpeg(path, sample_rate=sample_rate, mono=mono)
        except Exception:
            if not LIBROSA_AVAILABLE:
                raise

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        warnings.filterwarnings("ignore", message=".*__audioread_load.*", category=FutureWarning)
        audio, _ = librosa.load(path, sr=sample_rate, mono=mono)
    return audio


def _read_audio_soundfile(
    path: Path,
    sample_rate: int,
    mono: bool,
) -> npt.NDArray[np.float32]:
    audio, native_sr = sf.read(path, always_2d=False, dtype="float32")
    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if native_sr != sample_rate:
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa is required to resample audio")
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def _read_audio_ffmpeg(
    path: Path,
    sample_rate: int,
    mono: bool,
) -> npt.NDArray[np.float32]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-v", "error",
        "-nostdin",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
    ]
    if mono:
        command.extend(["-ac", "1"])
    command.append("-")
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio.copy()


def load_asvspoof_manifest(
    protocol_file: str | Path,
    audio_dir: str | Path,
) -> Iterator[tuple[Path, int]]:
    """Iterate over (audio_path, label) pairs from an ASVspoof protocol file.

    Supported examples (space-separated):
        ASVspoof 2019 LA protocol:
        SPEAKER_ID  UTTERANCE_ID  -  ATTACK_TYPE  LABEL

        ASVspoof 2021 LA CM trial metadata:
        SPEAKER_ID  UTTERANCE_ID  CODEC  TX  ATTACK  LABEL  TRIM  PARTITION

    The parser identifies the label token by value ('bonafide'/'spoof')
    instead of assuming a fixed column. Utterance IDs are resolved against
    the audio directory and support suffixed tokens such as
    `LA_E_1234567-alaw-ita_tx` by trimming trailing metadata when needed.

    Args:
        protocol_file: Path to the ASVspoof protocol .txt file
            (e.g., ASVspoof2019.LA.cm.train.trn.txt or trial_metadata.txt).
        audio_dir: Directory containing audio files.

    Yields:
        Tuples of (audio_path, label) where label is 0 (real) or 1 (fake).
    """
    protocol_file = Path(protocol_file)
    audio_dir = Path(audio_dir)

    with open(protocol_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue

            label = _extract_label(parts)
            if label is None:
                if line.strip().startswith("#"):
                    continue
                raise ValueError(
                    f"Could not find bonafide/spoof label in {protocol_file}:{line_no}: {line.strip()}"
                )

            audio_path = _resolve_audio_path(parts, audio_dir)
            yield audio_path, label


def _extract_label(parts: list[str]) -> int | None:
    """Return binary label inferred from any known label token."""
    for token in reversed(parts):
        normalized = token.strip().lower()
        if normalized in _LABEL_MAP:
            return _LABEL_MAP[normalized]
    return None


def _resolve_audio_path(parts: list[str], audio_dir: Path) -> Path:
    """Resolve utterance token to an existing audio path when possible."""
    candidate_tokens = []
    if len(parts) > 1:
        candidate_tokens.append(parts[1])
    candidate_tokens.extend(parts)

    seen = set()
    ordered_candidates = []
    for token in candidate_tokens:
        for normalized in _normalize_utterance_token(token):
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered_candidates.append(normalized)

    for token in ordered_candidates:
        direct = audio_dir / token
        if direct.exists():
            return direct
        for ext in _AUDIO_EXTENSIONS:
            with_ext = audio_dir / f"{token}{ext}"
            if with_ext.exists():
                return with_ext

    # Preserve legacy behavior if files are not locally materialized.
    default_token = parts[1] if len(parts) > 1 else parts[0]
    return audio_dir / f"{default_token}.flac"


def _normalize_utterance_token(token: str) -> list[str]:
    """Generate plausible utterance IDs from one protocol token."""
    token = token.strip()
    if not token:
        return []

    out = [token]
    stem = Path(token).stem
    if stem != token:
        out.append(stem)

    if "-" in token:
        out.append(token.split("-", 1)[0])
    if "-" in stem:
        out.append(stem.split("-", 1)[0])

    unique = []
    seen = set()
    for candidate in out:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique
