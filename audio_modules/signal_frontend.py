from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SignalFrontendConfig:
    fft_size: int = 1024
    hop_size: int = 256


class SignalFrontend:
    def __init__(self, config: SignalFrontendConfig | None = None):
        self.config = config or SignalFrontendConfig()

    def reset(self) -> None:
        return
