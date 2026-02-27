from __future__ import annotations

from typing import Callable

import numpy as np

from logging_utils import log_event


class AutoRanging:
    def __init__(self, config) -> None:
        self.config = config

        self._metric_peak_floor_enabled: bool = False
        self._energy_margin_history: list[float] = []
        self._energy_margin_target_low: float = 0.02
        self._energy_margin_target_high: float = 0.05
        self._energy_margin_adjustment_step: float = 0.002
        self._valley_history: list[float] = []
        self._valley_max_samples: int = 16
        self._prev_energy_for_valley: float = 0.0
        self._energy_was_falling: bool = False

        self._metric_audio_amp_enabled: bool = False
        self._audio_amp_check_interval_ms: float = 2500.0
        self._audio_amp_escalate_pct: float = 0.02
        self._last_audio_amp_check: float = 0.0
        self._audio_amp_hysteresis_count: int = 0
        self._metric_response_speed: float = float(getattr(config.auto_adjust, 'metric_response_speed', 1.0))

        self._metric_settled_threshold: int = 12
        self._metric_hysteresis_required: int = 2
        self._metric_settled_counts: dict[str, int] = {
            'peak_floor': 0,
            'sensitivity': 0,
            'audio_amp': 0,
        }
        self._metric_settled_flags: dict[str, bool] = {
            'peak_floor': False,
            'sensitivity': False,
            'audio_amp': False,
        }

    @property
    def valley_history(self) -> list[float]:
        return self._valley_history

    def observe_energy_for_valley(self, energy: float) -> None:
        if energy > self._prev_energy_for_valley and self._energy_was_falling:
            valley_val = self._prev_energy_for_valley
            if valley_val > 0.001:
                self._valley_history.append(valley_val)
                if len(self._valley_history) > self._valley_max_samples:
                    self._valley_history.pop(0)
        self._energy_was_falling = energy < self._prev_energy_for_valley
        self._prev_energy_for_valley = energy

    def reset_metric_settled_states(self) -> None:
        for key in self._metric_settled_counts:
            self._metric_settled_counts[key] = 0
            self._metric_settled_flags[key] = False

    def enable_metric_autoranging(self, metric: str, enable: bool = True) -> None:
        if metric == 'peak_floor':
            self._metric_peak_floor_enabled = enable
            if enable:
                self._energy_margin_history.clear()
                self._valley_history.clear()
                self._energy_was_falling = False
                self._metric_settled_counts['peak_floor'] = 0
                self._metric_settled_flags['peak_floor'] = False
                log_event("INFO", "MetricAutoRange", "Peak Floor metric enabled (valley-tracking)")
            else:
                log_event("INFO", "MetricAutoRange", "Peak Floor metric disabled")
        elif metric == 'audio_amp':
            self._metric_audio_amp_enabled = enable
            if enable:
                self._last_audio_amp_check = 0.0
                self._metric_settled_counts['audio_amp'] = 0
                self._metric_settled_flags['audio_amp'] = False
                log_event("INFO", "MetricAutoRange", "Audio Amp metric enabled (beat-driven)")
            else:
                log_event("INFO", "MetricAutoRange", "Audio Amp metric disabled")

    def set_metric_response_speed(self, speed: float) -> None:
        self._metric_response_speed = max(0.5, min(3.0, float(speed)))

    def _effective_metric_speed(self) -> float:
        return max(0.5, min(3.0, self._metric_response_speed))

    def _scaled_metric_interval_s(self, interval_ms: float) -> float:
        return (interval_ms / 1000.0) / self._effective_metric_speed()

    def _scaled_metric_step(self, base_step: float) -> float:
        return base_step * self._effective_metric_speed()

    def _effective_metric_hysteresis_required(self) -> int:
        speed = self._effective_metric_speed()
        if speed <= 1.0:
            return self._metric_hysteresis_required
        return max(1, int(round(self._metric_hysteresis_required / speed)))

    def _effective_metric_settled_threshold(self) -> int:
        speed = self._effective_metric_speed()
        return max(4, int(round(self._metric_settled_threshold / speed)));

    def get_metric_states(self) -> dict[str, str]:
        states = {}
        if self._metric_peak_floor_enabled:
            states['peak_floor'] = 'SETTLED' if self._metric_settled_flags.get('peak_floor', False) else 'ADJUSTING'

        if self._metric_audio_amp_enabled:
            states['audio_amp'] = 'SETTLED' if self._metric_settled_flags.get('audio_amp', False) else 'ADJUSTING'
        return states

    def compute_energy_margin_feedback(
        self,
        *,
        band_energy: float,
        peak_floor: float,
        audio_gain: float,
        callback: Callable | None = None,
    ) -> tuple[float, bool, int]:
        if not self._metric_peak_floor_enabled:
            return 0.0, False, 0

        if self._metric_settled_flags.get('peak_floor', False):
            margin = band_energy - peak_floor
            return margin, False, 0

        if len(self._valley_history) < 3:
            margin = band_energy - peak_floor
            self._energy_margin_history.append(margin)
            if len(self._energy_margin_history) > 16:
                self._energy_margin_history.pop(0)
            avg_margin = float(np.mean(self._energy_margin_history)) if self._energy_margin_history else margin
            return avg_margin, False, 0

        avg_valley = float(np.mean(self._valley_history))
        current_pf = peak_floor

        amp_floor = audio_gain * 0.10
        if avg_valley < amp_floor:
            avg_valley = amp_floor

        error = current_pf - avg_valley

        margin = band_energy - current_pf
        self._energy_margin_history.append(margin)
        if len(self._energy_margin_history) > 16:
            self._energy_margin_history.pop(0)

        tolerance = avg_valley * 0.20

        should_adjust = False
        direction = 0

        if error > tolerance:
            should_adjust = True
            direction = -1
        elif error < -tolerance:
            should_adjust = True
            direction = +1

        step = max(self._energy_margin_adjustment_step, avg_valley * 0.05)
        step = self._scaled_metric_step(step)

        if callback and should_adjust:
            self._metric_settled_counts['peak_floor'] = max(0, self._metric_settled_counts.get('peak_floor', 0) - 3)
            callback({
                'metric': 'peak_floor',
                'margin': float(np.mean(self._energy_margin_history)),
                'valley': avg_valley,
                'error': error,
                'adjustment': direction * step,
                'direction': 'raise' if direction > 0 else 'lower'
            })
        elif not should_adjust:
            self._metric_settled_counts['peak_floor'] = self._metric_settled_counts.get('peak_floor', 0) + 1
            if self._metric_settled_counts['peak_floor'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['peak_floor'] = True
                log_event("INFO", "Metric", "Peak Floor SETTLED", valley=f"{avg_valley:.4f}", pf=f"{current_pf:.4f}")

        return float(np.mean(self._energy_margin_history)), should_adjust, direction

    def compute_audio_amp_feedback(
        self,
        *,
        now: float,
        last_beat_time: float,
        beat_times: list[float],
        callback: Callable | None = None,
    ) -> None:
        if not self._metric_audio_amp_enabled:
            return

        if now - self._last_audio_amp_check < self._scaled_metric_interval_s(self._audio_amp_check_interval_ms):
            return
        self._last_audio_amp_check = now

        if self._metric_settled_flags.get('audio_amp', False):
            return

        from config import BEAT_RANGE_LIMITS

        amp_min, amp_max = BEAT_RANGE_LIMITS['audio_amp']
        amp_range = amp_max - amp_min
        step = amp_range * self._audio_amp_escalate_pct
        step = self._scaled_metric_step(step)

        time_since_beat = now - last_beat_time if last_beat_time > 0 else float('inf')
        ref_bps = 1.5
        target_interval = 1.0 / ref_bps

        wants_adjustment = bool(time_since_beat > target_interval * 3.0)

        wants_lower = False
        if last_beat_time > 0 and time_since_beat < target_interval and len(beat_times) >= 2:
            window_dur = beat_times[-1] - beat_times[0]
            if window_dur > 0:
                actual_bps = (len(beat_times) - 1) / window_dur
                if actual_bps > ref_bps * 2.0:
                    wants_lower = True

        if wants_adjustment or wants_lower:
            self._audio_amp_hysteresis_count += 1
            if self._audio_amp_hysteresis_count >= self._effective_metric_hysteresis_required():
                self._metric_settled_counts['audio_amp'] = max(0, self._metric_settled_counts.get('audio_amp', 0) - 3)
                self._audio_amp_hysteresis_count = 0
                if wants_lower:
                    lower_step = step * 0.5
                    if callback:
                        callback({
                            'metric': 'audio_amp',
                            'adjustment': -lower_step,
                            'direction': 'lower',
                            'reason': 'excess BPS > 2x reference (2x confirmed)',
                        })
                elif callback:
                    callback({
                        'metric': 'audio_amp',
                        'adjustment': +step,
                        'direction': 'raise',
                        'reason': f'no beats for {time_since_beat:.1f}s (2x confirmed)',
                    })
        else:
            self._audio_amp_hysteresis_count = 0
            self._metric_settled_counts['audio_amp'] = self._metric_settled_counts.get('audio_amp', 0) + 1
            if self._metric_settled_counts['audio_amp'] >= self._effective_metric_settled_threshold():
                self._metric_settled_flags['audio_amp'] = True
                log_event(
                    "INFO",
                    "Metric",
                    "Audio Amp SETTLED",
                    count=f"{self._metric_settled_counts['audio_amp']}",
                    threshold=f"{self._effective_metric_settled_threshold()}",
                )
