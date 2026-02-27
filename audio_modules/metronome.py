from __future__ import annotations

import numpy as np
import time

from logging_utils import log_event
from audio_modules.tempo_tracker import (
    build_acf_octave_candidates,
    dedup_window_seconds,
    effective_phase_accept_window_s,
    estimate_onset_bpm_from_times,
    metronome_phase_error_s,
    reference_bpm_for_onset_filters,
    select_acf_octave_candidate,
    within_dedup_window,
)


class MetronomeController:
    def __init__(self, engine) -> None:
        self.engine = engine

    def sync_tempo_tracker_state(self, tempo_locked: bool, is_downbeat: bool) -> None:
        eng = self.engine
        beat_phase = float(eng._metronome_phase % 1.0) if eng._metronome_bpm > 0 else 0.0
        eng._tempo_tracker.sync_runtime_state(
            metronome_bpm=eng._metronome_bpm,
            acf_confidence=eng._acf_confidence,
            tempo_locked=tempo_locked,
            phase_error_ms=eng.phase_error_ms,
            is_downbeat=is_downbeat,
            beat_phase=beat_phase,
        )

    def compute_tempo_lock_state(self, acf_confidence: float, downbeat_matches: int, now: float) -> bool:
        eng = self.engine
        conf = float(np.clip(acf_confidence, 0.0, 1.0))
        has_match = int(downbeat_matches) >= 1

        if not eng._tempo_lock_hysteresis_locked:
            enters = bool(
                conf >= eng._tempo_lock_enter_conf_base
                and (has_match or conf >= eng._tempo_lock_enter_conf_strict)
            )
            if enters:
                eng._tempo_lock_hysteresis_locked = True
                eng._tempo_lock_drop_started_at = 0.0
            return eng._tempo_lock_hysteresis_locked

        if conf <= eng._tempo_lock_exit_conf_base:
            if eng._tempo_lock_drop_started_at <= 0.0:
                eng._tempo_lock_drop_started_at = float(now)
            elif (float(now) - eng._tempo_lock_drop_started_at) >= eng._tempo_lock_exit_hold_s:
                eng._tempo_lock_hysteresis_locked = False
                eng._tempo_lock_drop_started_at = 0.0
        else:
            eng._tempo_lock_drop_started_at = 0.0

        return eng._tempo_lock_hysteresis_locked

    def reference_bpm_for_onset_filters(self) -> float:
        eng = self.engine
        return reference_bpm_for_onset_filters(
            eng._metronome_bpm,
            eng._acf_bpm_smoothed,
            eng.smoothed_tempo,
        )

    def effective_phase_accept_window_s(self) -> float:
        eng = self.engine
        return effective_phase_accept_window_s(
            eng._phase_accept_window_ms,
            eng._phase_accept_low_conf_mult,
            eng._acf_confidence,
        )

    def is_raw_onset_acceptable(self, now: float) -> bool:
        eng = self.engine
        bpm_ref = self.reference_bpm_for_onset_filters()
        dedup_window_s = dedup_window_seconds(bpm_ref, eng._beat_dedup_fraction, default_window_s=0.10)

        if within_dedup_window(eng._last_accepted_raw_onset_time, now, dedup_window_s):
            return False

        if eng._metronome_bpm > 0:
            phase_error_s = metronome_phase_error_s(eng._metronome_phase, eng._metronome_bpm)
            if phase_error_s > self.effective_phase_accept_window_s():
                return False

        return True

    def estimate_tempo_acf(self) -> None:
        eng = self.engine
        n = len(eng._onset_buffer)
        if n < 80:
            return

        signal = np.array(eng._onset_buffer, dtype=np.float64)
        signal = signal - np.mean(signal)

        n_fft = 1
        while n_fft < 2 * n:
            n_fft *= 2
        fft_sig = np.fft.rfft(signal, n=n_fft)
        acf = np.fft.irfft(fft_sig * np.conj(fft_sig))[:n]

        if acf[0] > 0:
            acf = acf / acf[0]
        else:
            return

        fps = eng._acf_onset_fps
        min_lag = max(1, int(fps * 60.0 / 200.0))
        max_lag = min(n - 1, int(fps * 60.0 / 55.0))
        if min_lag >= max_lag:
            return

        search = acf[min_lag:max_lag + 1]
        peak_idx = int(np.argmax(search))
        peak_value = float(search[peak_idx])

        if peak_value < 0.08:
            eng._acf_confidence = max(0.05, eng._acf_confidence * 0.9)
            return

        raw_lag = min_lag + peak_idx
        if peak_idx > 0 and peak_idx < len(search) - 1:
            alpha = float(search[peak_idx - 1])
            beta = float(search[peak_idx])
            gamma = float(search[peak_idx + 1])
            denom = alpha - 2.0 * beta + gamma
            correction = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-10 else 0.0
            refined_lag = raw_lag + correction
        else:
            refined_lag = float(raw_lag)

        bpm = 60.0 * fps / refined_lag

        candidates = build_acf_octave_candidates(
            bpm,
            peak_value,
            raw_lag,
            min_lag,
            max_lag,
            fps,
            acf,
        )

        target_bpm_hint = 0.0
        if eng._acf_bpm_smoothed > 0.0:
            target_bpm_hint = eng._acf_bpm_smoothed
        elif eng.smoothed_tempo > 0.0:
            target_bpm_hint = eng.smoothed_tempo
        bpm, peak_value, octave_mode, ranked_candidates = select_acf_octave_candidate(
            candidates,
            peak_value,
            eng._acf_confidence,
            eng._octave_target_bias_confidence_max,
            target_bpm_hint=target_bpm_hint,
        )
        if octave_mode == "target-guided" and ranked_candidates is not None:
            log_event("DEBUG", "ACF", "Octave disambig (target-guided)",
                      target=f"{target_bpm_hint:.0f}",
                      chosen=f"{bpm:.1f}",
                      candidates=str([(f"{c[0]:.1f}", f"{c[1]:.2f}") for c in ranked_candidates]))

        if bpm < 55 or bpm > 185:
            return

        eng._acf_confidence = float(peak_value)
        eng._acf_bpm = bpm

        smoothing = eng._tempo_tracker.smooth_acf_bpm_with_jump_gating(
            eng._acf_bpm_smoothed,
            bpm,
            peak_value,
            target_bpm_hint=target_bpm_hint,
        )
        eng._last_acf_smoothing_tag = smoothing.decision_tag
        eng._acf_bpm_smoothed = smoothing.smoothed_bpm

        if smoothing.decision_tag == "jump-target-validated":
            log_event("INFO", "ACF", "Tempo jump (target-validated)",
                      bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                      confidence=f"{peak_value:.3f}")
        elif smoothing.decision_tag == "jump-target-rejected":
            log_event("INFO", "ACF", "Tempo jump REJECTED (farther from target)",
                      bpm=f"{bpm:.1f}", target=f"{target_bpm_hint:.0f}",
                      current=f"{eng._acf_bpm_smoothed:.1f}")
        elif smoothing.decision_tag == "jump":
            log_event("INFO", "ACF", "Tempo jump",
                      bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}")
        elif smoothing.decision_tag == "initial":
            log_event("INFO", "ACF", "Initial tempo lock",
                      bpm=f"{bpm:.1f}", confidence=f"{peak_value:.3f}",
                      fps=f"{fps:.1f}")

    def estimate_onset_bpm(self) -> float:
        eng = self.engine
        return estimate_onset_bpm_from_times(
            eng._raw_onset_times,
            max_points=8,
            min_interval_s=0.15,
            max_interval_s=1.2,
            min_bpm=55.0,
            max_bpm=200.0,
        )

    def advance_metronome(self, now: float, band_energy: float = 0.0) -> None:
        eng = self.engine
        eng._metronome_beat_fired = False
        eng._metronome_downbeat_fired = False

        acf_conf = max(0.0, min(1.0, eng._acf_confidence))
        onset_bpm = self.estimate_onset_bpm()
        target_bpm = eng._tempo_tracker.update_from_acf_inputs(
            acf_confidence=acf_conf,
            onset_bpm=onset_bpm,
            acf_bpm_smoothed=eng._acf_bpm_smoothed,
            min_acf_weight=eng._tempo_fusion_min_acf_weight,
            max_acf_weight=eng._tempo_fusion_max_acf_weight,
        )

        if target_bpm <= 0 or (acf_conf < 0.10 and onset_bpm <= 0):
            if eng._metronome_bpm > 0:
                if eng._metronome_conf_lost_at <= 0:
                    eng._metronome_conf_lost_at = now
                hold_elapsed = now - eng._metronome_conf_lost_at
                if hold_elapsed <= eng._metronome_conf_hold_s:
                    decay_factor = max(0.0, 1.0 - (hold_elapsed / max(0.01, eng._metronome_conf_hold_s)) * 0.15)
                    target_bpm = eng._metronome_bpm * decay_factor
                else:
                    eng._metronome_bpm = 0.0
                    eng._metronome_conf_lost_at = 0.0
                    return
            else:
                eng._metronome_bpm = 0.0
                eng._metronome_conf_lost_at = 0.0
                return
        else:
            eng._metronome_conf_lost_at = 0.0

        if eng._metronome_bpm <= 0:
            eng._metronome_bpm = target_bpm
            eng._metronome_last_time = now
            eng._metronome_phase = 0.0
            eng._metronome_beat_count = 0
            log_event("INFO", "Metronome", "Started", bpm=f"{target_bpm:.1f}")
            return

        smoothing_conf = acf_conf if acf_conf > 0 else (0.20 if onset_bpm > 0 else 0.0)
        aggressive_ready = (
            eng._aggressive_tempo_snap_enabled
            and acf_conf >= eng._aggressive_snap_confidence
            and abs(eng.phase_error_ms) <= eng._aggressive_snap_phase_error_ms
            and eng.consecutive_matching_downbeats >= eng._aggressive_snap_min_matches
            and eng._metronome_bpm > 0
        )
        jump_ratio = abs(target_bpm - eng._metronome_bpm) / max(1e-6, eng._metronome_bpm)
        if aggressive_ready and jump_ratio <= eng._aggressive_snap_max_bpm_jump_ratio:
            eng._metronome_bpm = target_bpm
        else:
            alpha = eng._metronome_bpm_alpha_slow + (
                eng._metronome_bpm_alpha_fast - eng._metronome_bpm_alpha_slow
            ) * max(0.0, min(1.0, smoothing_conf))
            eng._metronome_bpm = (1.0 - alpha) * eng._metronome_bpm + alpha * target_bpm

        dt = now - eng._metronome_last_time
        eng._metronome_last_time = now
        if dt <= 0 or dt > 0.5:
            return

        eng._metronome_phase, crossings = eng._tempo_tracker.step_metronome_phase(
            eng._metronome_phase,
            eng._metronome_bpm,
            dt,
        )

        if crossings > 0:
            eng._metronome_beat_fired = True
            eng._metronome_beat_count += 1
            bpm = eng.beats_per_measure
            eng.beat_position_in_measure = (eng.beat_position_in_measure % bpm) + 1
            pos_idx = eng.beat_position_in_measure - 1

            decay = 0.85
            for i in range(bpm):
                eng.measure_energy_accum[i] *= decay
            eng.measure_energy_accum[pos_idx] += band_energy
            eng.measure_beat_counts[pos_idx] += 1

            avg_energies = []
            for i in range(bpm):
                if eng.measure_beat_counts[i] > 0:
                    avg_energies.append(eng.measure_energy_accum[i] / max(1.0, eng.measure_beat_counts[i]))
                else:
                    avg_energies.append(0.0)

            total_beats = sum(eng.measure_beat_counts)
            if total_beats >= bpm * 2:
                strongest_pos = int(np.argmax(avg_energies))
                mean_energy = np.mean(avg_energies) if np.mean(avg_energies) > 0 else 1.0
                eng.downbeat_confidence = avg_energies[strongest_pos] / mean_energy
                eng.downbeat_position = strongest_pos

            is_energy_downbeat = (pos_idx == eng.downbeat_position) and total_beats >= bpm * 2
            if is_energy_downbeat and eng.downbeat_pattern_enabled and eng._metronome_bpm > 0:
                pattern_matches = eng._validate_downbeat_against_pattern(now, use_bpm=eng._metronome_bpm)
                eng._metronome_downbeat_fired = pattern_matches
                eng.is_downbeat = pattern_matches

                if pattern_matches:
                    eng.consecutive_matching_downbeats += 1
                    log_event("INFO", "Downbeat", "Metronome+Energy accepted",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{eng.downbeat_confidence:.2f}",
                              consecutive=f"{eng.consecutive_matching_downbeats}/{eng.consecutive_match_threshold}",
                              error_ms=f"{eng.phase_error_ms:.1f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")
                    if abs(eng.phase_error_ms) > 5.0:
                        beat_period_ms = 60000.0 / eng._metronome_bpm
                        phase_correction = (eng.phase_error_ms / beat_period_ms) * 0.50
                        phase_correction = max(-0.20, min(0.20, phase_correction))
                        eng._metronome_phase += phase_correction
                        log_event("INFO", "Downbeat", "Phase correction from downbeat",
                                  error_ms=f"{eng.phase_error_ms:.1f}",
                                  correction=f"{phase_correction:.4f}")
                else:
                    eng.consecutive_matching_downbeats = max(0, eng.consecutive_matching_downbeats - 1)
                    eng._metronome_downbeat_fired = False
                    eng.is_downbeat = False
                    log_event("INFO", "Downbeat", "Metronome+Energy rejected",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{eng.downbeat_confidence:.2f}",
                              consecutive=f"{eng.consecutive_matching_downbeats}/{eng.consecutive_match_threshold}",
                              error_ms=f"{eng.phase_error_ms:.1f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")
            else:
                eng._metronome_downbeat_fired = is_energy_downbeat
                eng.is_downbeat = is_energy_downbeat
                if is_energy_downbeat:
                    log_event("INFO", "Downbeat", "Energy downbeat (metronome)",
                              position=f"{pos_idx+1}/{bpm}",
                              confidence=f"{eng.downbeat_confidence:.2f}",
                              energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]")

            eng._syncopation.on_metronome_beat()
            eng._syncopation_had_offbeat = eng._syncopation.had_offbeat
            eng._syncopation_streak = eng._syncopation.streak
            eng._syncopation_confirmed = eng._syncopation.confirmed
            eng._syncopation_armed = eng._syncopation.armed

            src = "DB" if eng._metronome_downbeat_fired else "bt"
            log_event("INFO", "Metronome", f"Tick [{src}]",
                      beat=f"{((eng._metronome_beat_count - 1) % bpm) + 1}/{bpm}",
                      bpm=f"{eng._metronome_bpm:.1f}",
                      acf_conf=f"{eng._acf_confidence:.2f}")

    def nudge_metronome_phase(self, onset_strength: float) -> None:
        eng = self.engine
        if eng._metronome_bpm <= 0:
            return

        phase_frac = eng._metronome_phase % 1.0
        error = -phase_frac if phase_frac < 0.5 else (1.0 - phase_frac)

        if abs(error) < eng._metronome_pll_window:
            conf = max(0.0, min(1.0, eng._acf_confidence))
            gain = eng._metronome_pll_base_gain + eng._metronome_pll_conf_gain * conf
            error_scale = 0.65 + 0.35 * min(1.0, abs(error) / 0.20)
            correction = error * gain * min(1.0, onset_strength) * error_scale
            correction = max(-0.22, min(0.22, correction))
            eng._metronome_phase += correction

    def reset_acf_metronome(self) -> None:
        eng = self.engine
        eng._onset_buffer.clear()
        eng._onset_callback_count = 0
        eng._onset_first_time = 0.0
        eng._fps_calibration_times.clear()
        eng._acf_bpm = 0.0
        eng._acf_bpm_smoothed = 0.0
        eng._acf_confidence = 0.0
        eng._metronome_phase = 0.0
        eng._metronome_beat_count = 0
        eng._metronome_conf_lost_at = 0.0
        eng._tempo_lock_hysteresis_locked = False
        eng._tempo_lock_drop_started_at = 0.0
        eng._metronome_bpm = 0.0
        eng._metronome_beat_fired = False
        eng._metronome_downbeat_fired = False
        eng._metronome_last_beat_time = 0.0
        log_event("INFO", "ACF", "Metronome reset")

    def update_tempo_tracking(self, current_time: float, energy: float = 0.0) -> None:
        eng = self.engine
        if not eng.tempo_tracking_enabled:
            return

        prev_beat_time = eng.last_beat_time
        eng.last_beat_time = current_time

        if prev_beat_time <= 0:
            return

        interval = current_time - prev_beat_time

        min_bpm = 60.0
        max_bpm = 200.0
        min_interval = 60.0 / max_bpm
        max_interval = 60.0 / min_bpm

        if interval > 0:
            raw_bpm = 60.0 / interval
            if raw_bpm < min_bpm or raw_bpm > max_bpm:
                log_event(
                    "INFO",
                    "Tempo",
                    "Tempo out of range",
                    bpm=f"{raw_bpm:.1f}",
                    min_bpm=f"{min_bpm:.1f}",
                    max_bpm=f"{max_bpm:.1f}",
                )
                return

        if interval < min_interval or interval > max_interval:
            log_event("INFO", "Tempo", "Interval rejected", interval=f"{interval:.3f}s", bpm=f"{60.0/interval:.1f}")
            return
        if interval <= 0.2:
            return

        if len(eng.beat_intervals) > 0:
            avg_interval = np.mean(eng.beat_intervals)
            if len(eng.beat_intervals) <= 3:
                lo_mult, hi_mult = 0.35, 2.8
            else:
                lo_mult, hi_mult = 0.5, 2.0
            if interval < (lo_mult * avg_interval) or interval > (hi_mult * avg_interval):
                log_event("INFO", "Tempo", "Outlier interval rejected", interval=f"{interval:.3f}s", avg=f"{avg_interval:.3f}s")
                return

        if eng.smoothed_tempo > 0 and eng.phase_snap_weight > 0 and eng.beat_stability > 0.3:
            predicted_interval = 60.0 / eng.smoothed_tempo
            if abs(interval - predicted_interval) / predicted_interval < 0.2:
                old_interval = interval
                interval = interval * (1 - eng.phase_snap_weight) + predicted_interval * eng.phase_snap_weight
                log_event("INFO", "Tempo", "Phase snap", old=f"{old_interval:.3f}s", new=f"{interval:.3f}s", predicted=f"{predicted_interval:.3f}s")

        eng.beat_intervals.append(interval)
        eng.beat_times.append(current_time)
        if len(eng.beat_intervals) > 16:
            eng.beat_intervals.pop(0)
            eng.beat_times.pop(0)

        weights = np.linspace(0.5, 1.5, len(eng.beat_intervals))
        weighted_avg_interval = float(np.average(eng.beat_intervals, weights=weights))
        new_tempo = float(60.0 / weighted_avg_interval) if weighted_avg_interval > 0 else 0.0

        smoothing_factor = 0.7
        if eng.smoothed_tempo > 0:
            smoothed_tempo = (smoothing_factor * eng.smoothed_tempo) + ((1 - smoothing_factor) * new_tempo)
            eng.smoothed_tempo = float(smoothed_tempo)
        else:
            eng.smoothed_tempo = float(new_tempo)

        if len(eng.beat_intervals) >= 3:
            intervals_arr = np.array(eng.beat_intervals)
            cv = np.std(intervals_arr) / np.mean(intervals_arr) if np.mean(intervals_arr) > 0 else 1.0
            eng.beat_stability = float(max(0.0, 1.0 - (cv / eng.stability_threshold)))

            if cv < eng.stability_threshold:
                eng.stable_tempo = eng.smoothed_tempo
                log_event("INFO", "Tempo", "Stable BPM committed", bpm=f"{eng.stable_tempo:.1f}", cv=f"{cv:.3f}", stability=f"{eng.beat_stability:.2f}")
            else:
                log_event("INFO", "Tempo", "BPM unstable", bpm=f"{eng.smoothed_tempo:.1f}", cv=f"{cv:.3f}", stability=f"{eng.beat_stability:.2f}")
        else:
            eng.beat_stability = 0.0

        eng.last_known_tempo = eng.smoothed_tempo
        self.predict_next_beat(current_time)

        metronome_active = (eng._acf_metronome_enabled and eng._metronome_bpm > 0)
        if metronome_active:
            return

        eng.beat_position_in_measure = (eng.beat_position_in_measure % eng.beats_per_measure) + 1
        pos_idx = eng.beat_position_in_measure - 1

        decay = 0.85
        for i in range(eng.beats_per_measure):
            eng.measure_energy_accum[i] *= decay
        eng.measure_energy_accum[pos_idx] += energy
        eng.measure_beat_counts[pos_idx] += 1

        avg_energies = []
        for i in range(eng.beats_per_measure):
            if eng.measure_beat_counts[i] > 0:
                avg_energies.append(eng.measure_energy_accum[i] / max(1.0, eng.measure_beat_counts[i]))
            else:
                avg_energies.append(0.0)

        total_beats = sum(eng.measure_beat_counts)
        if total_beats >= eng.beats_per_measure * 2:
            strongest_pos = int(np.argmax(avg_energies))
            mean_energy = np.mean(avg_energies) if np.mean(avg_energies) > 0 else 1.0
            eng.downbeat_confidence = avg_energies[strongest_pos] / mean_energy
            eng.downbeat_position = strongest_pos

        is_energy_downbeat = (pos_idx == eng.downbeat_position) and total_beats >= eng.beats_per_measure * 2

        if is_energy_downbeat and eng.downbeat_pattern_enabled and eng.smoothed_tempo > 0:
            pattern_matches = self.validate_downbeat_against_pattern(current_time, use_bpm=eng.smoothed_tempo)
            eng.is_downbeat = pattern_matches

            if pattern_matches:
                eng.consecutive_matching_downbeats += 1
                log_event(
                    "INFO",
                    "Downbeat",
                    "Accepted (raw)",
                    position=f"{pos_idx+1}/{eng.beats_per_measure}",
                    confidence=f"{eng.downbeat_confidence:.2f}",
                    consecutive=f"{eng.consecutive_matching_downbeats}/{eng.consecutive_match_threshold}",
                    error_ms=f"{eng.phase_error_ms:.1f}",
                    energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                )
            else:
                eng.consecutive_matching_downbeats = max(0, eng.consecutive_matching_downbeats - 1)
                log_event(
                    "INFO",
                    "Downbeat",
                    "Rejected (raw)",
                    position=f"{pos_idx+1}/{eng.beats_per_measure}",
                    confidence=f"{eng.downbeat_confidence:.2f}",
                    error_ms=f"{eng.phase_error_ms:.1f}",
                    energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                )
        else:
            eng.is_downbeat = is_energy_downbeat
            if eng.is_downbeat:
                log_event(
                    "INFO",
                    "Downbeat",
                    "Energy downbeat (raw)",
                    position=f"{pos_idx+1}/{eng.beats_per_measure}",
                    confidence=f"{eng.downbeat_confidence:.2f}",
                    energies="[" + ", ".join(f"{e:.2f}" for e in avg_energies) + "]"
                )

    def predict_next_beat(self, current_time: float, current_wall_time: float = 0.0) -> None:
        eng = self.engine
        wall_time = current_wall_time if current_wall_time > 0 else time.time()
        if eng._acf_metronome_enabled and eng._metronome_bpm > 0:
            phase_frac = eng._metronome_phase % 1.0
            beats_to_next = 1.0 - phase_frac if phase_frac > 1e-9 else 1.0
            predicted_interval = beats_to_next * (60.0 / eng._metronome_bpm)
            eng.predicted_next_beat_mono = current_time + predicted_interval
            eng.predicted_next_beat = wall_time + predicted_interval
            return

        if eng.smoothed_tempo > 0:
            predicted_interval = 60.0 / eng.smoothed_tempo
            eng.predicted_next_beat_mono = current_time + predicted_interval
            eng.predicted_next_beat = wall_time + predicted_interval

    def validate_downbeat_against_pattern(self, current_time: float, use_bpm: float = 0.0) -> bool:
        eng = self.engine
        active_bpm = use_bpm if use_bpm > 0 else eng.smoothed_tempo
        if active_bpm <= 0:
            return False

        beat_interval = 60.0 / active_bpm
        measure_interval = beat_interval * eng.beats_per_measure

        if eng.last_predicted_downbeat_time <= 0:
            eng.last_predicted_downbeat_time = current_time
            eng.consecutive_matching_downbeats = 1
            eng.phase_error_ms = 0.0
            return True

        predicted_time = eng.last_predicted_downbeat_time + measure_interval

        time_since_last = current_time - eng.last_predicted_downbeat_time
        if time_since_last > 0 and measure_interval > 0:
            measures_elapsed = round(time_since_last / measure_interval)
            if measures_elapsed >= 1:
                predicted_time = eng.last_predicted_downbeat_time + measures_elapsed * measure_interval

        eng.phase_error_ms = (current_time - predicted_time) * 1000.0

        effective_tolerance = eng.pattern_match_tolerance_ms
        if eng.consecutive_matching_downbeats < 2:
            effective_tolerance *= 1.5

        if abs(eng.phase_error_ms) <= effective_tolerance:
            eng.last_predicted_downbeat_time = current_time
            return True

        if abs(eng.phase_error_ms) <= effective_tolerance * 2.0:
            eng.last_predicted_downbeat_time = current_time
        return False

    def reset_downbeat_pattern(self) -> None:
        eng = self.engine
        eng.consecutive_matching_downbeats = 0
        eng.last_predicted_downbeat_time = 0.0
        eng.phase_error_ms = 0.0
        eng._auto_ranging.reset_metric_settled_states()

    def get_tempo_info(self) -> dict:
        eng = self.engine
        tempo_state = eng._tempo_tracker.get_state()
        reported_metronome_bpm = tempo_state.metronome_bpm if tempo_state.metronome_bpm > 0 else eng._metronome_bpm
        reported_acf_confidence = tempo_state.acf_confidence if tempo_state.acf_confidence > 0 else eng._acf_confidence
        reported_phase_error_ms = tempo_state.phase_error_ms if abs(tempo_state.phase_error_ms) > 1e-9 else eng.phase_error_ms
        reported_is_downbeat = bool(tempo_state.is_downbeat) or bool(eng.is_downbeat)

        display_bpm = eng.stable_tempo if eng.stable_tempo > 0 else eng.smoothed_tempo
        acf_active = eng._acf_metronome_enabled and reported_metronome_bpm > 0
        if acf_active:
            display_bpm = reported_metronome_bpm
            beat_pos = ((eng._metronome_beat_count - 1) % eng.beats_per_measure) + 1 if eng._metronome_beat_count > 0 else 0
        else:
            beat_pos = eng.beat_position_in_measure

        return {
            'bpm': display_bpm,
            'raw_bpm': eng.smoothed_tempo,
            'stable_bpm': eng.stable_tempo,
            'beat_position': beat_pos,
            'is_downbeat': reported_is_downbeat,
            'predicted_next_beat': eng.predicted_next_beat,
            'predicted_next_beat_mono': eng.predicted_next_beat_mono,
            'interval_count': len(eng.beat_intervals),
            'confidence': min(1.0, len(eng.beat_intervals) / 4.0),
            'stability': eng.beat_stability,
            'consecutive_matching_downbeats': eng.consecutive_matching_downbeats,
            'phase_error_ms': reported_phase_error_ms,
            'acf_bpm': eng._acf_bpm_smoothed,
            'acf_confidence': reported_acf_confidence,
            'acf_active': acf_active,
            'metronome_bpm': reported_metronome_bpm,
        }
