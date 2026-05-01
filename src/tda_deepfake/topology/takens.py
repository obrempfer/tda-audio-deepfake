"""Takens / time-delay embeddings on scalar audio signals."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from ..config import AudioConfig, SpectrogramConfig, TakensConfig
from ..features.extraction import build_raw_mel_spectrogram


def build_takens_signal(
    audio: npt.NDArray[np.float32],
    sample_rate: int = AudioConfig.SAMPLE_RATE,
    signal_type: str = TakensConfig.SIGNAL_TYPE,
    lowpass_cutoff_hz: Optional[float] = TakensConfig.LOWPASS_CUTOFF_HZ,
    filter_order: int = TakensConfig.FILTER_ORDER,
    signal_normalization: str = TakensConfig.SIGNAL_NORMALIZATION,
    envelope_compression: str = TakensConfig.ENVELOPE_COMPRESSION,
    envelope_smooth_sigma: float = TakensConfig.ENVELOPE_SMOOTH_SIGMA,
    n_mels: int = SpectrogramConfig.N_MELS,
    power: float = SpectrogramConfig.POWER,
    fmin: float = SpectrogramConfig.FMIN,
    fmax: Optional[float] = SpectrogramConfig.FMAX,
    band_split_low: float = SpectrogramConfig.BAND_SPLIT_LOW,
) -> npt.NDArray[np.float64]:
    """Construct one scalar signal for Takens embedding."""
    if audio.ndim != 1:
        raise ValueError(f"Takens signal expects a mono 1-D waveform, got shape {audio.shape}")

    signal_type = signal_type.lower()
    if signal_type == "low_wave":
        signal = _lowpass_waveform(
            audio,
            sample_rate=sample_rate,
            cutoff_hz=_resolve_lowpass_cutoff(
                sample_rate=sample_rate,
                cutoff_hz=lowpass_cutoff_hz,
                band_split_low=band_split_low,
            ),
            filter_order=filter_order,
        )
    elif signal_type == "full_wave":
        signal = np.asarray(audio, dtype=np.float64)
    elif signal_type in {"low_env", "full_env"}:
        signal = _mel_energy_envelope(
            audio,
            sample_rate=sample_rate,
            signal_type=signal_type,
            n_mels=n_mels,
            power=power,
            fmin=fmin,
            fmax=fmax,
            band_split_low=band_split_low,
            compression=envelope_compression,
            smooth_sigma=envelope_smooth_sigma,
        )
    else:
        raise ValueError(f"Unknown Takens signal type: {signal_type!r}")

    return _normalize_signal(np.asarray(signal, dtype=np.float64), method=signal_normalization)


def build_takens_embedding(
    signal: npt.NDArray[np.float64],
    embedding_dim: int = TakensConfig.EMBEDDING_DIM,
    delay: int = TakensConfig.DELAY,
    stride: int = TakensConfig.STRIDE,
) -> npt.NDArray[np.float64]:
    """Build a Takens delay embedding from one scalar signal."""
    if signal.ndim != 1:
        raise ValueError(f"Takens embedding expects a 1-D signal, got shape {signal.shape}")
    if embedding_dim < 2:
        raise ValueError(f"embedding_dim must be >= 2, got {embedding_dim}")
    if delay <= 0:
        raise ValueError(f"delay must be positive, got {delay}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    decimated = np.asarray(signal[::stride], dtype=np.float64)
    window = (embedding_dim - 1) * delay
    n_points = decimated.shape[0] - window
    if n_points <= 0:
        raise ValueError(
            "Signal too short for Takens embedding: "
            f"len={decimated.shape[0]} embedding_dim={embedding_dim} delay={delay}"
        )

    row_offsets = np.arange(n_points, dtype=np.int64)[:, None]
    col_offsets = (np.arange(embedding_dim, dtype=np.int64) * delay)[None, :]
    return decimated[row_offsets + col_offsets]


def _resolve_lowpass_cutoff(
    sample_rate: int,
    cutoff_hz: Optional[float],
    band_split_low: float,
) -> float:
    """Choose a low-band cutoff from config or the current low-band split."""
    nyquist = 0.5 * float(sample_rate)
    if cutoff_hz is None:
        cutoff_hz = nyquist * float(band_split_low)
    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0:
        raise ValueError(f"lowpass cutoff must be positive, got {cutoff_hz}")
    return min(cutoff_hz, nyquist * 0.99)


def _lowpass_waveform(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    cutoff_hz: float,
    filter_order: int,
) -> npt.NDArray[np.float64]:
    """Extract a low-band waveform with a Butterworth low-pass filter."""
    if filter_order <= 0:
        raise ValueError(f"filter_order must be positive, got {filter_order}")

    nyquist = 0.5 * float(sample_rate)
    normalized_cutoff = float(cutoff_hz) / nyquist
    if normalized_cutoff >= 1.0:
        return np.asarray(audio, dtype=np.float64)

    sos = butter(filter_order, normalized_cutoff, btype="lowpass", output="sos")
    return sosfiltfilt(sos, np.asarray(audio, dtype=np.float64))


def _mel_energy_envelope(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    signal_type: str,
    n_mels: int,
    power: float,
    fmin: float,
    fmax: Optional[float],
    band_split_low: float,
    compression: str,
    smooth_sigma: float,
) -> npt.NDArray[np.float64]:
    """Project a mel spectrogram to a 1-D energy envelope."""
    grid = build_raw_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        power=power,
        fmin=fmin,
        fmax=fmax,
    )
    if signal_type == "low_env":
        low_rows = max(1, int(np.ceil(grid.shape[0] * float(band_split_low))))
        grid = grid[:low_rows]

    envelope = np.sum(np.maximum(grid, 0.0), axis=0, dtype=np.float64)
    if compression == "none":
        pass
    elif compression == "log1p":
        envelope = np.log1p(envelope)
    else:
        raise ValueError(f"Unknown envelope compression: {compression!r}")

    if smooth_sigma > 0:
        envelope = gaussian_filter1d(envelope, sigma=float(smooth_sigma), mode="nearest")
    return np.asarray(envelope, dtype=np.float64)


def _normalize_signal(signal: npt.NDArray[np.float64], method: str) -> npt.NDArray[np.float64]:
    """Normalize a 1-D Takens signal before delay embedding."""
    if method == "none":
        return signal
    if method != "zscore":
        raise ValueError(f"Unknown Takens signal normalization: {method!r}")

    mean = float(np.mean(signal))
    std = float(np.std(signal))
    if std == 0.0:
        std = 1.0
    return (signal - mean) / std
