from __future__ import annotations

import numpy as np

from logging_utils import log_event


class SyncopationDetector:
    def __init__(self, config) -> None:
        self.config = config
        self.detected: bool = False
        self.window: float = float(getattr(config.beat, 'syncopation_window', 0.18))
        self.any_band_onset: bool = False
        self.streak: int = 0
        self.had_offbeat: bool = False
        self.confirmed: bool = False
        self.armed: bool = False

    @staticmethod
    def _texture_factor(combo_texture: float) -> float:
        texture = float(np.clip(combo_texture, -2.0, 3.0))
        if texture >= 1.0:
            return float(1.0 + ((texture - 1.0) / 2.0) * 1.0)
        return float(1.0 - ((1.0 - texture) / 3.0) * 0.5)

    def _effective_window(self, combo_texture: float) -> float:
        factor = self._texture_factor(combo_texture)
        return float(np.clip(self.window * factor, 0.05, 0.45))

    def update_any_band_onset(self, band_zscore_signals: dict[str, int], sync_band: str) -> bool:
        if sync_band == 'any':
            self.any_band_onset = any(signal == 1 for signal in band_zscore_signals.values())
        else:
            self.any_band_onset = band_zscore_signals.get(sync_band, 0) == 1
        return self.any_band_onset

    def process_frame(
        self,
        *,
        silence_veto_active: bool,
        sync_enabled: bool,
        metronome_bpm: float,
        metronome_phase: float,
        metronome_beat_fired: bool,
        bpm_limit: float,
        combo_texture: float,
    ) -> bool:
        self.detected = False
        if silence_veto_active:
            return False
        if not sync_enabled or metronome_bpm <= 0 or not self.any_band_onset or metronome_beat_fired:
            return False
        if metronome_bpm > bpm_limit:
            return False

        phase_frac = float(metronome_phase % 1.0)
        window = self._effective_window(combo_texture)
        dist_to_half = abs(phase_frac - 0.5)

        if dist_to_half >= window:
            return False

        self.had_offbeat = True
        if self.streak >= 1:
            self.detected = True
            log_event(
                "INFO",
                "Syncopation",
                "Off-beat onset detected",
                phase=f"{phase_frac:.2f}",
                bpm=f"{metronome_bpm:.1f}",
            )
        elif self.armed:
            self.detected = True
            self.streak = 1
            log_event(
                "INFO",
                "Syncopation",
                "Armed -> firing (2nd onset)",
                phase=f"{phase_frac:.2f}",
                bpm=f"{metronome_bpm:.1f}",
            )
        else:
            self.armed = True

        return self.detected

    def predictive_dropoff(
        self,
        *,
        metronome_bpm: float,
        metronome_phase: float,
        metronome_beat_fired: bool,
        combo_texture: float,
    ) -> None:
        if metronome_bpm <= 0 or metronome_beat_fired:
            return

        phase_frac = float(metronome_phase % 1.0)
        window = self._effective_window(combo_texture)
        if phase_frac > (0.5 + window) and not self.had_offbeat:
            if self.streak > 0 or self.armed:
                self.streak = 0
                self.confirmed = False
                self.armed = False
                log_event("INFO", "Syncopation", "Predictive drop-off (no onset in window)")

    def on_metronome_beat(self) -> None:
        if self.had_offbeat:
            self.streak += 1
        else:
            self.streak = 0
            self.confirmed = False
            self.armed = False

        self.had_offbeat = False
        if self.streak >= 1:
            self.confirmed = True
