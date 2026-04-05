from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import importlib
import importlib.util
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from audio_modules.contracts import FeatureFrame
from audio_modules.feature_extractors import (
    compute_bass_dominance,
    compute_multiband_energies,
    positive_spectral_flux,
)

_librosa_spec = importlib.util.find_spec("librosa")
if _librosa_spec is not None:  # pragma: no cover - optional dependency
    librosa = importlib.import_module("librosa")
    _HAS_LIBROSA = True
else:
    librosa = None
    _HAS_LIBROSA = False

_soundfile_spec = importlib.util.find_spec("soundfile")
if _soundfile_spec is not None:  # pragma: no cover - optional dependency
    sf = importlib.import_module("soundfile")
    _HAS_SOUNDFILE = True
else:
    sf = None
    _HAS_SOUNDFILE = False


_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sub_bass", 40.0, 200.0),
    ("low_mid", 200.0, 1000.0),
    ("mid", 1000.0, 3000.0),
    ("high", 3000.0, 8000.0),
)


@dataclass(slots=True)
class AnalysisConfig:
    sample_rate: int = 48000
    fft_size: int = 2048
    hop_size: int = 960
    window_size: int = 2208
    lowpass_enabled: bool = False
    lowpass_hz: float = 1000.0
    highpass_enabled: bool = False
    highpass_hz: float = 400.0
    freq_min_hz: float = 100.0
    freq_max_hz: float = 8000.0
    gain: float = 6.2


@dataclass(slots=True)
class AudioTimeline:
    samples: np.ndarray
    sample_rate: int
    duration_ms: int
    frame_times_ms: np.ndarray
    feature_frames: list[FeatureFrame]
    rms_per_frame: np.ndarray
    spectral_flux_per_frame: np.ndarray
    spectral_centroid_per_frame: np.ndarray
    spectral_flatness_per_frame: np.ndarray
    band_energies_per_frame: dict[str, np.ndarray]
    rms_mean_10s: np.ndarray
    rms_std_10s: np.ndarray
    flux_mean_10s: np.ndarray
    bass_mean_10s: np.ndarray
    energy_trend_10s: np.ndarray
    pitch_per_frame: np.ndarray
    pitch_confidence: np.ndarray
    p95_flux: float
    p95_band_energies: dict[str, float]


def _report(
    progress_callback: Callable[[str, float], None] | None,
    message: str,
    percent: float,
) -> None:
    if progress_callback is None:
        return
    progress_callback(message, float(percent))


def _as_mono_float32(samples: np.ndarray) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        return data
    if data.ndim == 2:
        mono = np.mean(data, axis=1)
        return np.asarray(mono, dtype=np.float32)
    return np.ravel(data).astype(np.float32, copy=False)


def _load_wav_builtin(path: Path) -> tuple[np.ndarray, int]:
    """Minimal WAV loader fallback for PCM content when external backends are unavailable."""
    with wave.open(str(path), "rb") as wf:
        channels = int(wf.getnchannels())
        sample_width = int(wf.getsampwidth())
        sr = int(wf.getframerate())
        frame_count = int(wf.getnframes())
        raw = wf.readframes(frame_count)

    if sample_width == 1:
        data_u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data_u8 - 128.0) / 128.0
    elif sample_width == 2:
        data_i16 = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data = data_i16 / 32768.0
    elif sample_width == 4:
        data_i32 = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data = data_i32 / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels)

    return _as_mono_float32(data), sr


def _resample_linear(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples.astype(np.float32, copy=False)
    if len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    duration_s = float(len(samples)) / float(orig_sr)
    target_len = max(1, int(round(duration_s * float(target_sr))))
    old_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False, dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64)
    out = np.interp(new_x, old_x, samples.astype(np.float64))
    return out.astype(np.float32)


def _align_frame_array(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if target_len <= 0:
        return np.array([], dtype=np.float32)
    if len(arr) == target_len:
        return arr
    if len(arr) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(arr) == 1:
        return np.full(target_len, float(arr[0]), dtype=np.float32)

    src_x = np.linspace(0.0, 1.0, num=len(arr), endpoint=True, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=True, dtype=np.float64)
    aligned = np.interp(dst_x, src_x, arr.astype(np.float64))
    return aligned.astype(np.float32)


def load_audio(
    file_path: str | Path,
    config: AnalysisConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    """Load audio file (or extract it from video) as mono float32."""
    source = Path(file_path)
    work_path = source
    temp_path: Path | None = None

    _report(progress_callback, "Loading audio...", 5.0)

    if source.suffix.lower() in {".mp4", ".mkv", ".avi", ".webm", ".wmv", ".mov", ".flv"}:
        _report(progress_callback, "Extracting audio from video...", 20.0)
        temp_path = extract_video_audio(source)
        work_path = temp_path

    try:
        if _HAS_SOUNDFILE:
            assert sf is not None
            data, sr = sf.read(work_path, dtype="float32", always_2d=False)
            samples = _as_mono_float32(np.asarray(data))
        elif _HAS_LIBROSA:
            assert librosa is not None
            samples, sr = librosa.load(work_path, sr=None, mono=True)
            samples = _as_mono_float32(samples)
        elif Path(work_path).suffix.lower() == ".wav":
            samples, sr = _load_wav_builtin(Path(work_path))
        else:
            raise RuntimeError(
                "Audio loading requires either soundfile or librosa. "
                "Install with: pip install soundfile librosa (or use WAV files)."
            )

        if int(sr) != int(config.sample_rate):
            if _HAS_LIBROSA:
                assert librosa is not None
                samples = librosa.resample(samples.astype(np.float32), orig_sr=int(sr), target_sr=int(config.sample_rate))
            else:
                samples = _resample_linear(samples, int(sr), int(config.sample_rate))
        _report(progress_callback, "Audio loaded", 100.0)
        return samples.astype(np.float32, copy=False)
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def extract_video_audio(video_path: str | Path) -> Path:
    """Extract mono 48kHz WAV from a video file using ffmpeg."""
    src = Path(video_path)
    fd, temp_name = tempfile.mkstemp(prefix="pmv_extract_", suffix=".wav")
    os.close(fd)
    out_path = Path(temp_name)

    ffmpeg_bin = _resolve_ffmpeg_binary()

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        "-ac",
        "1",
        str(out_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg executable could not be launched. Install ffmpeg or set FFMPEG_BINARY to a valid executable path."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg extraction failed: {proc.stderr.strip()}")
    return out_path


def _resolve_ffmpeg_binary() -> str:
    """Resolve ffmpeg from env, PATH, or imageio-ffmpeg fallback."""
    env_candidate = (os.environ.get("FFMPEG_BINARY") or "").strip()
    if env_candidate:
        if Path(env_candidate).exists():
            return str(Path(env_candidate))
        resolved_env = shutil.which(env_candidate)
        if resolved_env:
            return str(resolved_env)

    resolved = shutil.which("ffmpeg")
    if resolved:
        return str(resolved)

    imageio_spec = importlib.util.find_spec("imageio_ffmpeg")
    if imageio_spec is not None:  # pragma: no cover - optional dependency
        try:
            imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
            exe = str(imageio_ffmpeg.get_ffmpeg_exe())
            if exe and Path(exe).exists():
                return exe
        except Exception:
            pass

    raise FileNotFoundError(
        "ffmpeg not found. Add ffmpeg to PATH, set FFMPEG_BINARY, or install imageio-ffmpeg."
    )


def apply_frequency_filters(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    """Apply optional lowpass/highpass filtering to magnitude spectrum."""
    out = np.asarray(spectrum, dtype=np.float32).copy()

    if config.lowpass_enabled:
        out[np.asarray(freqs) > float(config.lowpass_hz)] = 0.0
    if config.highpass_enabled:
        out[np.asarray(freqs) < float(config.highpass_hz)] = 0.0
    return out


def extract_pitch(
    samples: np.ndarray,
    sr: int,
    hop_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pitch extraction via librosa.piptrack with weighted mean per frame."""
    if not _HAS_LIBROSA:
        est_len = max(1, int(np.ceil(len(samples) / max(1, hop_length))))
        return np.zeros(est_len, dtype=np.float32), np.zeros(est_len, dtype=np.float32)

    assert librosa is not None

    pitches, magnitudes = librosa.piptrack(
        y=np.asarray(samples, dtype=np.float32),
        sr=int(sr),
        hop_length=int(hop_length),
    )

    n_frames = pitches.shape[1] if pitches.ndim == 2 else 0
    if n_frames <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Vectorised weighted-mean pitch per frame (replaces per-frame Python loop)
    mag_sum = magnitudes.sum(axis=0)  # (n_frames,)
    safe_mag = np.where(mag_sum > 1e-12, mag_sum, 1.0)
    weighted_pitch = (pitches * magnitudes).sum(axis=0) / safe_mag
    pitch_values = np.where(
        (mag_sum > 1e-12) & (weighted_pitch > 0.0),
        np.log10(np.maximum(weighted_pitch, 1e-12)),
        0.0,
    ).astype(np.float32)
    confidence = mag_sum.astype(np.float32)

    conf_norm, _ = p95_normalize(confidence)
    return pitch_values.astype(np.float32), conf_norm.astype(np.float32)


def p95_normalize(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize values by their 95th percentile."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.copy(), 0.0

    p95 = float(np.percentile(arr, 95))
    if p95 <= 1e-9:
        return np.zeros_like(arr, dtype=np.float32), p95
    return (arr / p95).astype(np.float32), p95


def compute_rolling_aggregates(
    feature_frames: list[FeatureFrame],
    frame_times_ms: np.ndarray,
    rms_per_frame: np.ndarray,
    flux_per_frame: np.ndarray,
    band_energies: dict[str, np.ndarray],
    window_sec: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute centered 10-second rolling statistics for model features."""
    _ = feature_frames
    n = len(rms_per_frame)
    if n == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty, empty, empty

    if len(frame_times_ms) > 1:
        hop_ms = float(np.median(np.diff(frame_times_ms)))
    else:
        hop_ms = 20.0
    window_frames = max(1, int(round((float(window_sec) * 1000.0) / max(1e-6, hop_ms))))
    half = max(0, window_frames // 2)

    bass = np.asarray(band_energies.get("sub_bass", np.zeros(n, dtype=np.float32)), dtype=np.float32)

    # Vectorised rolling mean/std via uniform_filter instead of per-element slicing
    from scipy.ndimage import uniform_filter1d
    kern = max(1, 2 * half + 1)
    rms_mean = uniform_filter1d(rms_per_frame.astype(np.float64), size=kern, mode="nearest").astype(np.float32)
    flux_mean = uniform_filter1d(flux_per_frame.astype(np.float64), size=kern, mode="nearest").astype(np.float32)
    bass_mean = uniform_filter1d(bass.astype(np.float64), size=kern, mode="nearest").astype(np.float32)

    # Rolling std via E[x²] - E[x]²
    rms_sq_mean = uniform_filter1d((rms_per_frame.astype(np.float64)) ** 2, size=kern, mode="nearest")
    rms_std = np.sqrt(np.maximum(rms_sq_mean - rms_mean.astype(np.float64) ** 2, 0.0)).astype(np.float32)

    # Trend: slope of linear regression in each window — vectorised via rolling covariance
    rms_f64 = rms_per_frame.astype(np.float64)
    x_global = np.arange(n, dtype=np.float64)
    x_mean_local = uniform_filter1d(x_global, size=kern, mode="nearest")
    y_mean_local = uniform_filter1d(rms_f64, size=kern, mode="nearest")
    xy_mean = uniform_filter1d(x_global * rms_f64, size=kern, mode="nearest")
    x2_mean = uniform_filter1d(x_global ** 2, size=kern, mode="nearest")
    denom = x2_mean - x_mean_local ** 2
    trend = np.where(denom > 1e-12,
                     (xy_mean - x_mean_local * y_mean_local) / denom,
                     0.0).astype(np.float32)

    return rms_mean, rms_std, flux_mean, bass_mean, trend


def _extract_feature_bundle(
    samples: np.ndarray,
    config: AnalysisConfig,
    progress_callback: Callable[[str, float], None] | None,
) -> tuple[
    np.ndarray,
    list[FeatureFrame],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    float,
    dict[str, float],
]:
    sr = int(config.sample_rate)
    hop = max(1, int(config.hop_size))
    win = max(1, int(config.window_size))
    fft_size = max(win, int(config.fft_size))

    if len(samples) == 0:
        empty = np.array([], dtype=np.float32)
        return empty, [], empty, empty, empty, empty, {name: empty for name, _, _ in _BANDS}, 0.0, {
            name: 0.0 for name, _, _ in _BANDS
        }

    starts = list(range(0, max(len(samples) - win + 1, 1), hop))
    if starts:
        last_start = starts[-1]
        if last_start + win < len(samples):
            starts.append(len(samples) - win)
    else:
        starts = [0]

    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(fft_size, d=1.0 / float(sr)).astype(np.float32)

    n = len(starts)
    frame_times = np.zeros(n, dtype=np.float32)
    raw_rms = np.zeros(n, dtype=np.float32)
    raw_flux = np.zeros(n, dtype=np.float32)
    centroid = np.zeros(n, dtype=np.float32)
    flatness = np.zeros(n, dtype=np.float32)
    raw_bands: dict[str, np.ndarray] = {
        name: np.zeros(n, dtype=np.float32) for name, _, _ in _BANDS
    }

    prev_spectrum: np.ndarray | None = None

    for i, start in enumerate(starts):
        segment = samples[start:start + win]
        if len(segment) < win:
            padded = np.zeros(win, dtype=np.float32)
            padded[: len(segment)] = segment
            segment = padded
        frame_times[i] = float(start) * 1000.0 / float(sr)

        frame_rms = float(np.sqrt(np.mean(segment ** 2)))
        raw_rms[i] = frame_rms

        spec = np.abs(np.fft.rfft(segment * window, n=fft_size)).astype(np.float32)
        spec = apply_frequency_filters(spec, freqs, config)

        raw_flux[i] = float(positive_spectral_flux(prev_spectrum, spec, max_filter_size=1))
        prev_spectrum = spec

        total = float(np.sum(spec))
        if total > 1e-12:
            centroid[i] = float(np.sum(freqs * spec) / total)
            geo = float(np.exp(np.mean(np.log(np.maximum(spec, 1e-12)))))
            ari = float(np.mean(spec))
            flatness[i] = float(geo / max(ari, 1e-12))

        band_vals = compute_multiband_energies(spec, sr, config.gain, _BANDS)
        for band_name in raw_bands:
            raw_bands[band_name][i] = float(band_vals.get(band_name, 0.0))

        if i % max(1, n // 10) == 0:
            pct = 30.0 * float(i) / float(max(1, n - 1))
            _report(progress_callback, "Computing FFT frames...", pct)

    rms_norm, _ = p95_normalize(raw_rms)
    flux_norm, p95_flux = p95_normalize(raw_flux)
    band_norm: dict[str, np.ndarray] = {}
    p95_bands: dict[str, float] = {}
    for band_name, values in raw_bands.items():
        nvals, p95 = p95_normalize(values)
        band_norm[band_name] = nvals
        p95_bands[band_name] = float(p95)

    energy_delta = np.diff(rms_norm, prepend=rms_norm[0] if len(rms_norm) else 0.0).astype(np.float32)
    flux_delta = np.diff(flux_norm, prepend=flux_norm[0] if len(flux_norm) else 0.0).astype(np.float32)

    features: list[FeatureFrame] = []
    for i in range(n):
        sub = float(band_norm["sub_bass"][i])
        low = float(band_norm["low_mid"][i])
        mid = float(band_norm["mid"][i])
        high = float(band_norm["high"][i])
        total_band = max(1e-9, sub + low + mid + high)
        hfc_proxy = float(np.clip(high / total_band, 0.0, 1.0))
        features.append(
            FeatureFrame(
                flux_norm=float(flux_norm[i]),
                energy_norm=float(rms_norm[i]),
                energy_delta=float(energy_delta[i]),
                flux_delta=float(flux_delta[i]),
                hfc_proxy=hfc_proxy,
                sub_bass=sub,
                low_mid=low,
                mid=mid,
                high=high,
                bass_dominance=float(compute_bass_dominance(sub, low, mid, high)),
            )
        )

    _report(progress_callback, "Extracting bREadbeats features...", 50.0)
    return frame_times, features, rms_norm, flux_norm, centroid, flatness, band_norm, p95_flux, p95_bands


class OfflineFeatureExtractor:
    """Offline wrapper that produces FeatureFrame-compatible outputs."""

    def __init__(self, config: AnalysisConfig):
        self._config = config

    def process_full_file(
        self,
        samples: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
    ) -> list[FeatureFrame]:
        callback: Callable[[str, float], None] | None
        if progress_callback is None:
            callback = None
        else:
            callback = lambda _msg, pct: progress_callback(float(pct))

        _, features, *_ = _extract_feature_bundle(_as_mono_float32(samples), self._config, callback)
        return features


def analyze_full_file(
    samples: np.ndarray,
    config: AnalysisConfig,
    progress_callback: Callable[[str, float], None] | None = None,
) -> AudioTimeline:
    """Run offline end-to-end feature extraction for the full audio file."""
    mono = _as_mono_float32(samples)

    frame_times, features, rms_norm, flux_norm, centroid, flatness, band_norm, p95_flux, p95_bands = _extract_feature_bundle(
        mono,
        config,
        progress_callback,
    )

    _report(progress_callback, "Extracting pitch...", 50.0)
    pitch_raw, pitch_conf = extract_pitch(mono, config.sample_rate, config.hop_size)
    pitch_aligned = _align_frame_array(pitch_raw, len(features))
    pitch_conf_aligned = _align_frame_array(pitch_conf, len(features))

    # If pitch extraction yielded no useful data (e.g. librosa unavailable),
    # fall back to log10(spectral_centroid) which is already computed per frame
    # and serves as a reasonable pitch proxy (center-of-mass frequency).
    if pitch_aligned.size > 0 and float(np.max(np.abs(pitch_aligned))) < 1e-9:
        safe_centroid = np.maximum(centroid, 1.0)
        pitch_aligned = np.log10(safe_centroid).astype(np.float32)
        pitch_conf_aligned = np.ones(len(features), dtype=np.float32)

    _report(progress_callback, "Extracting pitch...", 70.0)

    _report(progress_callback, "Computing aggregates...", 70.0)
    rms_mean_10s, rms_std_10s, flux_mean_10s, bass_mean_10s, energy_trend_10s = compute_rolling_aggregates(
        features,
        frame_times,
        rms_norm,
        flux_norm,
        band_norm,
        window_sec=10.0,
    )

    _report(progress_callback, "Computing aggregates...", 85.0)
    _report(progress_callback, "Normalizing...", 100.0)

    duration_ms = int(round(len(mono) * 1000.0 / float(max(1, config.sample_rate))))
    return AudioTimeline(
        samples=mono,
        sample_rate=int(config.sample_rate),
        duration_ms=duration_ms,
        frame_times_ms=frame_times,
        feature_frames=features,
        rms_per_frame=rms_norm,
        spectral_flux_per_frame=flux_norm,
        spectral_centroid_per_frame=centroid,
        spectral_flatness_per_frame=flatness,
        band_energies_per_frame=band_norm,
        rms_mean_10s=rms_mean_10s,
        rms_std_10s=rms_std_10s,
        flux_mean_10s=flux_mean_10s,
        bass_mean_10s=bass_mean_10s,
        energy_trend_10s=energy_trend_10s,
        pitch_per_frame=pitch_aligned,
        pitch_confidence=pitch_conf_aligned,
        p95_flux=float(p95_flux),
        p95_band_energies=p95_bands,
    )
