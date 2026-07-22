"""
Offline orbital replay generator — *better-than-live* edition.

Replays the live StrokeMapper engine offline against PMV analysis data
to produce native alpha/beta funscript actions.  Because the full audio
is pre-analysed we can do things live cannot:

* **Perfect timing** — simulation clock replaces wall-clock so
  time.perf_counter() inside StrokeMapper/BeatIntelligence returns
  simulated time (fixes fill-silence-ramp & dBFS-reference-decay bugs).
* **100 Hz stepping** — 10 ms steps (live ≈ 25–40 ms callback jitter).
* **Sub-step beat delivery** — beats fire at their exact ms, not
  quantized to the step grid.
* **Perfect lookahead** — ``predicted_next_beat_mono`` is always the
  real next beat (live only has a noisy prediction).
* **Pre-warm** — 2 s of audio runs through the engine before output
  recording so silence-gate, RMS envelope, and z-score detectors are
  properly initialised.
* **Post-process** — beat-snapped peak alignment + tighter
  point-reduction tolerance.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from audio_modules.contracts import BeatEvent, FeatureFrame, RMS_DB_FLOOR, rms_to_dbfs
from audio_modules.feature_extractors import (
    compute_multiband_energies,
    estimate_dominant_frequency,
)
from config import Config
from pmv_audio_analysis import AudioTimeline, AnalysisConfig
from pmv_beat_engine import BeatTimeline
from pmv_funscript_io import FunscriptAction
from stroke_mapper import StrokeMapper


# ── Z-Score peak detector (offline clone of AudioEngine.ZScorePeakDetector) ──

class _ZScorePeakDetector:
    __slots__ = ('lag', 'threshold', 'influence', 'buffer', 'filtered',
                 'mean', 'std', 'initialized', '_buf_len')

    def __init__(self, lag: int = 30, threshold: float = 2.5, influence: float = 0.05):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.buffer: list[float] = []
        self.filtered: list[float] = []
        self.mean: float = 0.0
        self.std: float = 0.0
        self.initialized: bool = False
        self._buf_len: int = 0

    def update(self, value: float) -> int:
        self.buffer.append(value)
        self._buf_len += 1
        if self._buf_len < self.lag:
            self.filtered.append(value)
            return 0
        if not self.initialized:
            window = self.buffer[:self.lag]
            self.mean = float(np.mean(window))
            self.std = float(np.std(window))
            self.filtered.append(value)
            self.initialized = True
            return 0
        deviation = value - self.mean
        if self.std > 0 and abs(deviation) > self.threshold * self.std:
            signal = 1 if deviation > 0 else -1
            filtered_val = self.influence * value + (1 - self.influence) * self.filtered[-1]
        else:
            signal = 0
            filtered_val = value
        self.filtered.append(filtered_val)
        recent = self.filtered[-self.lag:]
        self.mean = float(np.mean(recent))
        self.std = float(np.std(recent))
        return signal


# ── Lightweight audio engine stub for offline replay ──

class _AudioEngineStub:
    """Provides the interface StrokeMapper / BeatIntelligence read from AudioEngine.

    Per-frame state is set by the replay loop before each process_beat() call.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._spectrum: Optional[np.ndarray] = None
        self._band_energies: dict[str, float] = {}
        self._band_zscore_signals: dict[str, int] = {
            'sub_bass': 0, 'low_mid': 0, 'mid': 0, 'high': 0,
        }
        self.predicted_next_beat_mono: float = 0.0
        self._metronome_bpm: float = 0.0
        self._metronome_phase: float = 0.0
        self.silence_gate_active: bool = False

    def get_spectrum(self) -> Optional[np.ndarray]:
        return self._spectrum

    def get_band_energies(self) -> dict[str, float]:
        return dict(self._band_energies)

    def get_live_fourphase_band_energies(self) -> dict[str, float]:
        return dict(self._band_energies)

    def set_silence_gate(self, active: bool) -> None:
        self.silence_gate_active = active

    def _estimate_frequency(
        self,
        spectrum: np.ndarray,
        low_hz: Optional[float] = None,
        high_hz: Optional[float] = None,
    ) -> float:
        return estimate_dominant_frequency(spectrum, self.sample_rate, low_hz, high_hz)


# ── Pre-computation of offline signals ──

# Use the same band boundaries as the PMV offline analysis pipeline
# (pmv_audio_analysis._BANDS) so that both offline generators — restim
# stroke-based and orbital replay — share a consistent spectral view.
_ZSCORE_BANDS: list[tuple[str, int, int]] = [
    ('sub_bass', 40, 200),
    ('low_mid', 200, 1000),
    ('mid', 1000, 3000),
    ('high', 3000, 8000),
]


@dataclass
class _FrameIndex:
    """Compact per-frame data — no spectrum stored (saves ~8 KB/frame)."""
    time_ms: float
    feature: FeatureFrame
    rms: float
    rms_db: float
    flux: float
    centroid_hz: float
    band_energies: dict[str, float] = field(default_factory=dict)
    zscore_signals: dict[str, int] = field(default_factory=dict)
    bass_dom_freq: float = 125.0
    full_dom_freq: float = 500.0
    sample_start: int = 0  # offset into raw samples for on-demand FFT


@dataclass
class _FFTParams:
    """Cached FFT parameters for on-demand spectrum computation."""
    samples: np.ndarray          # raw audio (float32)
    window: np.ndarray           # Hann window (float32)
    win_size: int
    fft_size: int
    sample_rate: int
    freq_bins: np.ndarray        # frequency axis for masking
    freq_mask: np.ndarray        # bool mask: True = keep bin


def _compute_spectrum_on_demand(fft_params: _FFTParams, frame: _FrameIndex) -> np.ndarray:
    """Compute FFT magnitude spectrum from raw samples for one frame.

    Applies the same frequency focus masking (lowpass / highpass /
    freq_min / freq_max) that the PMV analysis pipeline uses so that
    the orbital replay sees a consistent spectral picture.
    """
    p = fft_params
    start = frame.sample_start
    segment = p.samples[start:start + p.win_size]
    if len(segment) < p.win_size:
        padded = np.zeros(p.win_size, dtype=np.float32)
        padded[:len(segment)] = segment
        segment = padded
    spec = np.abs(np.fft.rfft(segment * p.window, n=p.fft_size)).astype(np.float32)
    spec[~p.freq_mask] = 0.0
    return spec


def _precompute_frame_index(
    timeline: AudioTimeline,
    analysis_cfg: AnalysisConfig,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[list[_FrameIndex], _FFTParams]:
    """Build compact per-frame index with z-score signals and dominant freqs.

    Uses the same band boundaries as the PMV offline analysis pipeline
    (40/200/1000/3000/8000) so both offline generators share a consistent
    spectral view.

    **Memory-efficient**: does NOT store spectrum arrays.  The simulation
    computes FFT on-demand only for frames it actually visits.
    For a 3-hour file this keeps memory usage at ~150 MB instead of ~8 GB.
    """
    sr = int(analysis_cfg.sample_rate)
    hop = max(1, int(analysis_cfg.hop_size))
    win = max(1, int(analysis_cfg.window_size))
    fft_size = max(win, int(analysis_cfg.fft_size))
    gain = float(analysis_cfg.gain)

    samples = np.asarray(timeline.samples, dtype=np.float32)
    n_frames = len(timeline.frame_times_ms)

    # Z-score detectors per band (same params as live engine)
    detectors = {
        name: _ZScorePeakDetector(lag=30, threshold=2.5, influence=0.05)
        for name, _, _ in _ZSCORE_BANDS
    }

    window = np.hanning(win).astype(np.float32)
    starts = list(range(0, max(len(samples) - win + 1, 1), hop))
    if starts and starts[-1] + win < len(samples):
        starts.append(len(samples) - win)
    if not starts:
        starts = [0]

    frames: list[_FrameIndex] = []
    report_every = max(1, min(n_frames, len(starts)) // 20)

    for i in range(min(n_frames, len(starts))):
        start = starts[i]
        segment = samples[start:start + win]
        if len(segment) < win:
            padded = np.zeros(win, dtype=np.float32)
            padded[:len(segment)] = segment
            segment = padded

        # Temporary spectrum for z-score / dom-freq computation only
        spec = np.abs(np.fft.rfft(segment * window, n=fft_size)).astype(np.float32)

        # Band energies with live engine boundaries
        band_energies_raw = compute_multiband_energies(spec, sr, gain, _ZSCORE_BANDS)

        # Feed z-score detectors
        zscore_signals: dict[str, int] = {}
        for name, _, _ in _ZSCORE_BANDS:
            energy = float(band_energies_raw.get(name, 0.0))
            sig = detectors[name].update(energy)
            zscore_signals[name] = sig

        # Dominant frequencies
        bass_dom = estimate_dominant_frequency(spec, sr, 30.0, 500.0)
        if bass_dom <= 0.0:
            bass_dom = 125.0
        full_dom = estimate_dominant_frequency(spec, sr, 80.0, 8000.0)
        if full_dom <= 0.0:
            full_dom = 500.0

        # RMS
        raw_rms = float(timeline.rms_per_frame[i]) if i < len(timeline.rms_per_frame) else 0.0
        raw_rms_db = rms_to_dbfs(raw_rms) if raw_rms > 0.0 else RMS_DB_FLOOR

        frames.append(_FrameIndex(
            time_ms=float(timeline.frame_times_ms[i]),
            feature=timeline.feature_frames[i] if i < len(timeline.feature_frames) else FeatureFrame(),
            rms=raw_rms,
            rms_db=raw_rms_db,
            flux=float(timeline.spectral_flux_per_frame[i]) if i < len(timeline.spectral_flux_per_frame) else 0.0,
            centroid_hz=float(timeline.spectral_centroid_per_frame[i]) if i < len(timeline.spectral_centroid_per_frame) else 0.0,
            band_energies=band_energies_raw,
            zscore_signals=zscore_signals,
            bass_dom_freq=bass_dom,
            full_dom_freq=full_dom,
            sample_start=start,
        ))

        # spec is not stored — GC reclaims it
        if progress_callback and i % report_every == 0:
            progress_callback("Building frame index...", 2.0 + 6.0 * (i / max(1, n_frames)))

    freq_bins = np.fft.rfftfreq(fft_size, d=1.0 / sr).astype(np.float32)
    freq_mask = np.ones(len(freq_bins), dtype=bool)
    freq_mask[freq_bins < float(analysis_cfg.freq_min_hz)] = False
    freq_mask[freq_bins > float(analysis_cfg.freq_max_hz)] = False
    if analysis_cfg.lowpass_enabled:
        freq_mask[freq_bins > float(analysis_cfg.lowpass_hz)] = False
    if analysis_cfg.highpass_enabled:
        freq_mask[freq_bins < float(analysis_cfg.highpass_hz)] = False

    fft_params = _FFTParams(
        samples=samples,
        window=window,
        win_size=win,
        fft_size=fft_size,
        sample_rate=sr,
        freq_bins=freq_bins,
        freq_mask=freq_mask,
    )
    return frames, fft_params


# ── Beat event synthesis ──

def _find_frame_index(frame_times_ms: np.ndarray, target_ms: float) -> int:
    """Binary search for nearest frame index at or before target_ms."""
    idx = int(np.searchsorted(frame_times_ms, target_ms, side='right')) - 1
    return max(0, min(idx, len(frame_times_ms) - 1))


def _build_beat_event(
    frame: _FrameIndex,
    is_beat: bool,
    is_downbeat: bool,
    is_syncopated: bool,
    bpm: float,
    beat_band: str,
    fired_bands: list[str],
    mono_time: float,
    intensity: float,
    metronome_bpm: float,
    beat_features: Optional[dict] = None,
) -> BeatEvent:
    """Construct a BeatEvent from offline frame data."""
    return BeatEvent(
        timestamp=mono_time,
        intensity=intensity,
        frequency=frame.full_dom_freq,
        is_beat=is_beat,
        spectral_flux=frame.flux,
        peak_energy=frame.rms,
        is_downbeat=is_downbeat,
        bpm=bpm,
        tempo_reset=False,
        tempo_locked=bpm > 0,
        phase_error_ms=0.0,
        beat_band=beat_band,
        fired_bands=fired_bands,
        metronome_bpm=metronome_bpm,
        acf_confidence=0.8 if bpm > 0 else 0.0,
        is_syncopated=is_syncopated,
        monotonic_timestamp=mono_time,
        beat_features=beat_features,
        spectral_centroid_hz=frame.centroid_hz,
        spectral_flatness=0.0,
        raw_rms=frame.rms,
        raw_rms_db=frame.rms_db,
    )


def _compute_beat_features(frame: _FrameIndex) -> dict:
    """Compute beat_features dict from offline frame data."""
    f = frame.feature
    low = max(0.0, float(f.sub_bass) + float(f.low_mid))
    high_val = max(0.0, float(f.high))
    hfc = float(np.clip(f.hfc_proxy, 0.0, 1.0))
    body = float(np.clip(f.energy_norm, 0.0, 1.0))
    attack = float(np.clip(f.flux_delta, 0.0, 1.0))

    kick_like = float(np.clip((0.45 * low) + (0.30 * body) + (0.25 * (1.0 - hfc)), 0.0, 1.0))
    hat_like = float(np.clip((0.45 * high_val) + (0.35 * hfc) + (0.20 * attack), 0.0, 1.0))
    mixed_conf = float(np.clip(1.0 - abs(kick_like - hat_like), 0.0, 1.0))

    # Z-score boost (same as live engine)
    sigs = frame.zscore_signals
    if sigs.get('sub_bass', 0) == 1 or sigs.get('low_mid', 0) == 1:
        kick_like = max(kick_like, 0.75)
    if sigs.get('high', 0) == 1:
        hat_like = max(hat_like, 0.75)

    return {
        'kick_like_conf': kick_like,
        'hat_like_conf': hat_like,
        'mixed_conf': mixed_conf,
        'bass_dominance': float(np.clip(f.bass_dominance, 0.0, 8.0)),
        'new_beat_score': 0.5,
        'new_raw_onset_conf': 0.5,
        'bus_scores': {},
        'bus_pass': {},
        'bus_reason_codes': {},
        'frontend_ms': 0.0,
        'tempo_ms': 0.0,
        'detector_ms': 0.0,
        'sidecar_ms': 0.0,
    }


# ── Main replay function ──

@dataclass
class OrbitalReplayResult:
    alpha_actions: list[FunscriptAction]
    beta_actions: list[FunscriptAction]
    duration_ms: int


def _find_next_beat(beat_times_arr: np.ndarray, current_ms: float, start_idx: int) -> int:
    """Return index of the first beat strictly after *current_ms*, or len(arr)."""
    idx = start_idx
    while idx < len(beat_times_arr) and beat_times_arr[idx] <= current_ms:
        idx += 1
    return idx


# ── Simulation-time monkey-patch ──────────────────────────────────────────

class _SimClock:
    """Thread-local replacement for ``time.perf_counter`` during replay.

    Stores the current simulation time (seconds, matching monotonic_timestamp
    domain) so that any unconditional ``time.perf_counter()`` call inside
    StrokeMapper / BeatIntelligence returns simulation time instead of
    wall-clock.  This fixes:
      * fill-silence-ramp mismatch (stroke_mapper line 1169)
      * dBFS-reference-decay mismatch (beat_intelligence line 890)
    """
    __slots__ = ('_t',)

    def __init__(self) -> None:
        self._t: float = 0.0

    def set(self, t: float) -> None:
        self._t = t

    def __call__(self) -> float:          # drop-in for time.perf_counter
        return self._t


# ── Post-process: beat-snap peaks ─────────────────────────────────────────

def _beat_snap_peaks(
    actions: list[FunscriptAction],
    beat_times_ms: np.ndarray,
    snap_window_ms: float = 30.0,
) -> list[FunscriptAction]:
    """Shift local extrema to the nearest beat if within *snap_window_ms*.

    This compensates for any residual grid-quantisation from the 10 ms
    step rate.  A peak is an action whose pos is a local max or local min
    compared to its immediate neighbours.
    """
    if len(actions) < 3 or len(beat_times_ms) == 0:
        return actions

    out = list(actions)
    for i in range(1, len(out) - 1):
        prev_p, cur_p, next_p = out[i - 1].pos, out[i].pos, out[i + 1].pos
        is_peak = (cur_p > prev_p and cur_p > next_p) or (cur_p < prev_p and cur_p < next_p)
        if not is_peak:
            continue
        cur_at = out[i].at
        # Find nearest beat
        bi = int(np.searchsorted(beat_times_ms, cur_at, side='right'))
        candidates: list[float] = []
        if bi < len(beat_times_ms):
            candidates.append(float(beat_times_ms[bi]))
        if bi > 0:
            candidates.append(float(beat_times_ms[bi - 1]))
        for bt in candidates:
            if abs(bt - cur_at) <= snap_window_ms:
                out[i] = FunscriptAction(at=int(round(bt)), pos=cur_p)
                break
    return out


def replay_orbital(
    timeline: AudioTimeline,
    beat_timeline: BeatTimeline,
    config: Config,
    analysis_cfg: AnalysisConfig,
    step_ms: float = 10.0,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> OrbitalReplayResult:
    """Replay the live orbital engine offline — **better-than-live** edition.

    Improvements over a naive 1:1 replay:

    1. **Simulation clock** — ``time.perf_counter`` is monkey-patched for
       the duration of this call so every internal wall-clock read returns
       simulation time.  Fixes fill-silence-ramp and dBFS-reference-decay
       mismatches.
    2. **100 Hz stepping** — *step_ms* defaults to 10 ms (live ≈ 25–40 ms).
    3. **Sub-step beat delivery** — when a beat falls inside a step
       window, the event is delivered with the *exact* beat timestamp
       (not the step-grid time), and a second non-beat event is issued
       at the grid time so fill/expression keeps advancing.
    4. **Perfect lookahead** — ``predicted_next_beat_mono`` always points
       at the real next beat (live only has a noisy estimator).
    5. **2 s pre-warm** — the first 2 000 ms are replayed but not
       recorded so silence-gate, RMS envelope, z-score detectors, and
       band-energy EMAs are properly initialised.
    6. **Post-process** — peaks are snapped to the nearest beat, then
       point-reduction runs with a tighter tolerance (0.8 vs 1.0).
    """

    def _report(msg: str, pct: float) -> None:
        if progress_callback is not None:
            progress_callback(msg, pct)

    _report("Preparing offline replay...", 0.0)

    # ── Build compact frame index (no spectrums stored) ──
    frames, fft_params = _precompute_frame_index(timeline, analysis_cfg, _report)
    if not frames:
        return OrbitalReplayResult([], [], int(timeline.duration_ms))

    _report("Pre-computing audio features...", 5.0)

    frame_times = np.array([f.time_ms for f in frames], dtype=np.float64)
    duration_ms = int(timeline.duration_ms)
    sr = int(analysis_cfg.sample_rate)

    # ── Build beat lookup ──
    beat_times_ms: list[float] = []
    beat_types: list[str] = []
    beat_confidences: list[float] = []

    for b in beat_timeline.beats:
        beat_times_ms.append(float(b.time_ms))
        beat_types.append(str(b.beat_type) if b.beat_type else "beat")
        beat_confidences.append(float(b.confidence))

    beat_times_arr = np.array(beat_times_ms, dtype=np.float64) if beat_times_ms else np.array([], dtype=np.float64)

    bpm = float(beat_timeline.tempo_bpm) if beat_timeline.tempo_bpm else 120.0

    # ── Set up stub audio engine ──
    stub = _AudioEngineStub(sample_rate=sr)
    stub._metronome_bpm = bpm

    # ── Instantiate StrokeMapper ──
    mapper = StrokeMapper(
        config=config,
        get_volume=lambda: 1.0,
        audio_engine=stub,
    )

    # ── Install simulation clock ──
    sim_clock = _SimClock()
    _real_perf_counter = time.perf_counter
    time.perf_counter = sim_clock  # type: ignore[assignment]

    try:
        return _run_simulation(
            mapper, stub, sim_clock,
            frames, fft_params, frame_times,
            beat_times_arr, beat_types, beat_confidences,
            bpm, sr, duration_ms, step_ms,
            _report,
        )
    finally:
        # Always restore real clock, even on error
        time.perf_counter = _real_perf_counter  # type: ignore[assignment]


def _run_simulation(
    mapper: StrokeMapper,
    stub: _AudioEngineStub,
    sim_clock: _SimClock,
    frames: list[_FrameIndex],
    fft_params: _FFTParams,
    frame_times: np.ndarray,
    beat_times_arr: np.ndarray,
    beat_types: list[str],
    beat_confidences: list[float],
    bpm: float,
    sr: int,
    duration_ms: int,
    step_ms: float,
    _report: Callable[[str, float], None],
) -> OrbitalReplayResult:

    mono_base = 1000.0  # arbitrary monotonic base (avoids zero)
    primary_band = 'sub_bass'

    # ── Pre-warm: run 2 s without recording ──
    PREWARM_MS = 2000.0
    _report("Pre-warming engine (2 s)...", 8.0)

    total_steps = int(math.ceil(duration_ms / step_ms))
    prewarm_steps = int(math.ceil(min(PREWARM_MS, duration_ms) / step_ms))
    report_interval = max(1, total_steps // 80)

    beat_idx = 0
    prev_alpha = 0.0
    prev_beta = 0.0

    def _step_engine(t_ms_: float, recording: bool) -> tuple[float, float]:
        """Run one simulation step.  Returns (alpha, beta) in [-1, 1]."""
        nonlocal beat_idx, prev_alpha, prev_beta

        mono_time = mono_base + t_ms_ / 1000.0
        sim_clock.set(mono_time)

        fi = _find_frame_index(frame_times, t_ms_)
        frame = frames[fi]

        # Compute spectrum on-demand from raw samples (not stored)
        spectrum = _compute_spectrum_on_demand(fft_params, frame)

        # Update stub
        stub._spectrum = spectrum
        stub._band_energies = dict(frame.band_energies)
        stub._band_zscore_signals = dict(frame.zscore_signals)
        stub._metronome_bpm = bpm

        # Perfect lookahead
        next_bi = _find_next_beat(beat_times_arr, t_ms_, beat_idx)
        if next_bi < len(beat_times_arr):
            stub.predicted_next_beat_mono = mono_base + beat_times_arr[next_bi] / 1000.0
        else:
            stub.predicted_next_beat_mono = 0.0

        # ── Sub-step: deliver beats at exact timestamps ──
        # Collect beats in this window [t_ms_, t_ms_ + step_ms)
        beats_in_window: list[tuple[float, int]] = []  # (exact_ms, beat_list_idx)
        bi_scan = beat_idx
        window_end = t_ms_ + step_ms
        while bi_scan < len(beat_times_arr) and beat_times_arr[bi_scan] < window_end:
            beats_in_window.append((float(beat_times_arr[bi_scan]), bi_scan))
            bi_scan += 1

        alpha = prev_alpha
        beta = prev_beta

        if beats_in_window:
            # Fire each beat at its exact timestamp
            for exact_ms, b_idx in beats_in_window:
                exact_mono = mono_base + exact_ms / 1000.0
                sim_clock.set(exact_mono)

                bi_frame = _find_frame_index(frame_times, exact_ms)
                b_frame = frames[bi_frame]
                b_spectrum = _compute_spectrum_on_demand(fft_params, b_frame)
                stub._spectrum = b_spectrum
                stub._band_energies = dict(b_frame.band_energies)
                stub._band_zscore_signals = dict(b_frame.zscore_signals)

                bt = beat_types[b_idx]
                beat_features = _compute_beat_features(b_frame)
                fired = [n for n, s in b_frame.zscore_signals.items() if s == 1]

                event = _build_beat_event(
                    frame=b_frame,
                    is_beat=True,
                    is_downbeat=(bt == "downbeat"),
                    is_syncopated=(bt == "syncopation"),
                    bpm=bpm,
                    beat_band=primary_band,
                    fired_bands=fired,
                    mono_time=exact_mono,
                    intensity=max(float(np.clip(b_frame.rms * 3.0, 0.0, 1.0)),
                                  beat_confidences[b_idx]),
                    metronome_bpm=bpm,
                    beat_features=beat_features,
                )
                cmd = mapper.process_beat(event)
                if cmd is not None:
                    alpha = float(np.clip(cmd.alpha, -1.0, 1.0))
                    beta = float(np.clip(cmd.beta, -1.0, 1.0))

                if recording:
                    a_pos = max(0, min(100, int(round((alpha + 1.0) * 50.0))))
                    b_pos = max(0, min(100, int(round((beta + 1.0) * 50.0))))
                    _recorded_alpha.append(FunscriptAction(at=int(round(exact_ms)), pos=a_pos))
                    _recorded_beta.append(FunscriptAction(at=int(round(exact_ms)), pos=b_pos))

            beat_idx = bi_scan

            # Issue a trailing non-beat event at the grid time so
            # fill/expression keeps advancing between beats.
            sim_clock.set(mono_time)
            stub._spectrum = spectrum
            stub._band_energies = dict(frame.band_energies)
            stub._band_zscore_signals = dict(frame.zscore_signals)

            event = _build_beat_event(
                frame=frame,
                is_beat=False,
                is_downbeat=False,
                is_syncopated=False,
                bpm=bpm,
                beat_band=primary_band,
                fired_bands=[n for n, s in frame.zscore_signals.items() if s == 1],
                mono_time=mono_time,
                intensity=float(np.clip(frame.rms * 3.0, 0.0, 1.0)),
                metronome_bpm=bpm,
            )
            cmd = mapper.process_beat(event)
            if cmd is not None:
                alpha = float(np.clip(cmd.alpha, -1.0, 1.0))
                beta = float(np.clip(cmd.beta, -1.0, 1.0))
        else:
            # No beats — just a fill/expression frame
            event = _build_beat_event(
                frame=frame,
                is_beat=False,
                is_downbeat=False,
                is_syncopated=False,
                bpm=bpm,
                beat_band=primary_band,
                fired_bands=[n for n, s in frame.zscore_signals.items() if s == 1],
                mono_time=mono_time,
                intensity=float(np.clip(frame.rms * 3.0, 0.0, 1.0)),
                metronome_bpm=bpm,
            )
            cmd = mapper.process_beat(event)
            if cmd is not None:
                alpha = float(np.clip(cmd.alpha, -1.0, 1.0))
                beta = float(np.clip(cmd.beta, -1.0, 1.0))

        prev_alpha = alpha
        prev_beta = beta
        return alpha, beta

    # ── Pre-warm pass (no recording) ──
    _recorded_alpha: list[FunscriptAction] = []
    _recorded_beta: list[FunscriptAction] = []

    for pw_i in range(prewarm_steps):
        t_ms = pw_i * step_ms
        _step_engine(t_ms, recording=False)

    _report("Running orbital replay (100 Hz)...", 12.0)

    # ── Main recording pass ──
    for step_i in range(total_steps):
        t_ms = step_i * step_ms
        if t_ms > duration_ms:
            break

        alpha, beta = _step_engine(t_ms, recording=True)

        # Record grid-point sample (beat sub-steps were already recorded)
        a_pos = max(0, min(100, int(round((alpha + 1.0) * 50.0))))
        b_pos = max(0, min(100, int(round((beta + 1.0) * 50.0))))
        at_ms = int(round(t_ms))
        _recorded_alpha.append(FunscriptAction(at=at_ms, pos=a_pos))
        _recorded_beta.append(FunscriptAction(at=at_ms, pos=b_pos))

        if step_i % report_interval == 0:
            pct = 12.0 + 78.0 * (step_i / max(1, total_steps))
            _report("Running orbital replay (100 Hz)...", pct)

    _report("Post-processing orbital output...", 92.0)

    # ── De-duplicate: sort by time, keep last value at each timestamp ──
    def _dedup(actions: list[FunscriptAction]) -> list[FunscriptAction]:
        actions.sort(key=lambda a: a.at)
        if not actions:
            return actions
        out: list[FunscriptAction] = [actions[0]]
        for a in actions[1:]:
            if a.at == out[-1].at:
                out[-1] = a  # keep latest value
            else:
                out.append(a)
        return out

    alpha_actions = _dedup(_recorded_alpha)
    beta_actions = _dedup(_recorded_beta)

    # ── Beat-snap peaks ──
    _report("Snapping peaks to beats...", 94.0)
    if len(beat_times_arr) > 0:
        alpha_actions = _beat_snap_peaks(alpha_actions, beat_times_arr, snap_window_ms=25.0)
        beta_actions = _beat_snap_peaks(beta_actions, beat_times_arr, snap_window_ms=25.0)

    # ── Simplify with tighter tolerance (0.8 vs 1.0) ──
    _report("Simplifying orbital output...", 96.0)
    alpha_actions = _simplify_actions(alpha_actions, tolerance=0.8)
    beta_actions = _simplify_actions(beta_actions, tolerance=0.8)

    _report("Orbital replay complete", 100.0)

    return OrbitalReplayResult(
        alpha_actions=alpha_actions,
        beta_actions=beta_actions,
        duration_ms=duration_ms,
    )


def _simplify_actions(
    actions: list[FunscriptAction],
    tolerance: float = 1.0,
) -> list[FunscriptAction]:
    """Remove intermediate points that don't deviate from linear interpolation.

    Keeps first, last, and any point where the interpolated value differs
    from the actual value by more than `tolerance` funscript units.
    """
    if len(actions) <= 2:
        return list(actions)

    keep = [True] * len(actions)
    i = 0
    while i < len(actions) - 2:
        j = i + 1
        while j < len(actions) - 1:
            # Check if point j can be removed (interpolation from i to j+1 is close enough)
            a_i = actions[i]
            a_next = actions[j + 1]
            dt_total = a_next.at - a_i.at
            if dt_total <= 0:
                j += 1
                continue
            # Check all intermediate points between i and j+1
            can_remove = True
            for k in range(i + 1, j + 1):
                a_k = actions[k]
                dt_k = a_k.at - a_i.at
                t = dt_k / dt_total
                interp = a_i.pos + t * (a_next.pos - a_i.pos)
                if abs(a_k.pos - interp) > tolerance:
                    can_remove = False
                    break
            if can_remove:
                keep[j] = False
                j += 1
            else:
                break
        i = j

    return [a for a, k in zip(actions, keep) if k]
