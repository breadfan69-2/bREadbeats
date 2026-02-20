# Config Runtime Read Audit

Generated: 2026-02-20

Method: instrumented config dataclass field reads while running selected tests in-process via pytest.

Pytest exit code: 0

Schema keys: 266
Read at runtime: 259
Unread at runtime: 21

## Selected test workload

- tests/test_phase1_foundations.py
- tests/test_phase2_readiness_silence.py
- tests/test_phase3_gates.py
- tests/test_phase5_learning.py
- tests/test_phase6_bpm_jitter.py
- tests/test_stroke_mapper_contract.py
- tests/test_network_lifecycle.py
- tests/test_config_persistence.py
- tests/test_close_persist_wiring.py

## Unread keys during this runtime session

- `auto_adjust.enabled_params.audio_amp`
- `auto_adjust.enabled_params.flux_mult`
- `auto_adjust.enabled_params.peak_decay`
- `auto_adjust.enabled_params.peak_floor`
- `auto_adjust.enabled_params.rise_sens`
- `auto_adjust.enabled_params.sensitivity`
- `stroke.orbit_geometry.beat.center_y`
- `stroke.orbit_geometry.beat.max_radius`
- `stroke.orbit_geometry.beat.park_radius`
- `stroke.orbit_geometry.creep.center_y`
- `stroke.orbit_geometry.creep.max_radius`
- `stroke.orbit_geometry.creep.park_radius`
- `stroke.orbit_geometry.downbeat.center_y`
- `stroke.orbit_geometry.downbeat.max_radius`
- `stroke.orbit_geometry.downbeat.park_radius`
- `stroke.orbit_geometry.start.center_y`
- `stroke.orbit_geometry.start.max_radius`
- `stroke.orbit_geometry.start.park_radius`
- `stroke.orbit_geometry.syncopation.center_y`
- `stroke.orbit_geometry.syncopation.max_radius`
- `stroke.orbit_geometry.syncopation.park_radius`
