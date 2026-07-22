import time

import numpy as np
from funscript_converter import (
    LAYOUT_MODEL_ALIASES,
    LAYOUT_MODELS,
    constrain_fourphase_coordinates,
    decode_layout_controls,
    tetrahedral_project,
)
from network_engine import TCodeCommand


_LIVE_FOURPHASE_CENTER = np.ones(4, dtype=np.float64)
_CLASSIC_RING_ELECTRODE_ANGLES = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=np.float64)
_DIRECT_FOURPHASE_TAGS = ("E1", "E2", "E3", "E4")
LIVE_FOURPHASE_STRAIGHT_LAYOUT = "Straight Line"
LIVE_FOURPHASE_LAYOUT_OPTIONS = (
    LIVE_FOURPHASE_STRAIGHT_LAYOUT,
    "Pair At Top",
    "Pair At Middle",
    "Pair At Bottom / Rear",
)
LIVE_FOURPHASE_BAND_OPTIONS = (
    "sub_bass",
    "bass",
    "low_mid",
    "mid",
    "upper_mid",
    "presence",
    "brilliance",
)
LIVE_FOURPHASE_BEAT_RESPONSE_CURVE_OPTIONS = ("linear", "ease", "bell")
DEFAULT_LIVE_FOURPHASE_BAND_MAPPING: tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]
] = (
    ("mid", "upper_mid", "presence"),
    ("low_mid", "mid"),
    ("bass", "low_mid"),
    ("sub_bass", "bass"),
)
DEFAULT_LIVE_FOURPHASE_BEAT_RESPONSE_CURVES: tuple[str, str, str, str] = (
    "linear",
    "linear",
    "linear",
    "linear",
)
DEFAULT_LIVE_FOURPHASE_BEAT_RADIUS_CONTRAST_STRENGTH = 0.0
DEFAULT_LIVE_FOURPHASE_BEAT_SPEED_SPREAD_STRENGTH = 0.0
DEFAULT_LIVE_FOURPHASE_TETRA_POST_PROJECTION_EXPANSION = 1.0
DEFAULT_LIVE_FOURPHASE_BANDROUTER_FILL_MIX = 0.12
DEFAULT_LIVE_FOURPHASE_BANDROUTER_IDLE_FLOOR = 0.10
DEFAULT_LIVE_FOURPHASE_BANDROUTER_POST_PROJECTION_EXPANSION = 1.0
DEFAULT_LIVE_FOURPHASE_BANDROUTER_PULSE_INTERVAL_RANDOM_PERCENT = 10.0
LIVE_FOURPHASE_BANDROUTER_MAX_BRILLIANCE_RANDOM_PERCENT = 50.0
_BANDROUTER_SPECTRAL_WEIGHT = 0.85


def normalize_live_fourphase_model(model: str | None) -> str:
    normalized = str(model or "tetra3d").strip().lower()
    if normalized == "classic":
        return "classic"
    if normalized in {"bandrouter", "band_routed", "band-routed", "band routed"}:
        return "bandrouter"
    return "tetra3d"


def normalize_live_fourphase_band_name(band: str | None) -> str:
    normalized = str(band or "sub_bass").strip().lower()
    if normalized == "high":
        normalized = "presence"
    return normalized if normalized in LIVE_FOURPHASE_BAND_OPTIONS else "sub_bass"


def normalize_live_fourphase_beat_response_curve(curve: str | None) -> str:
    normalized = str(curve or "linear").strip().lower()
    return normalized if normalized in LIVE_FOURPHASE_BEAT_RESPONSE_CURVE_OPTIONS else "linear"


def normalize_live_fourphase_beat_response_curves(
    curves: object | None,
) -> tuple[str, str, str, str]:
    default_curves = DEFAULT_LIVE_FOURPHASE_BEAT_RESPONSE_CURVES
    if not isinstance(curves, (list, tuple)) or len(curves) != 4:
        return default_curves

    normalized_curves: list[str] = []
    for index, default_curve in enumerate(default_curves):
        raw_curve = curves[index] if index < len(curves) else default_curve
        normalized_curves.append(normalize_live_fourphase_beat_response_curve(raw_curve))

    return tuple(normalized_curves)  # type: ignore[return-value]


def normalize_live_fourphase_beat_radius_contrast_strength(value: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BEAT_RADIUS_CONTRAST_STRENGTH if value is None else value,
            0.0,
            1.0,
        )
    )


def normalize_live_fourphase_beat_speed_spread_strength(value: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BEAT_SPEED_SPREAD_STRENGTH if value is None else value,
            0.0,
            1.0,
        )
    )


def normalize_live_fourphase_tetra_post_projection_expansion(value: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_TETRA_POST_PROJECTION_EXPANSION if value is None else value,
            0.0,
            2.0,
        )
    )


def normalize_live_fourphase_bandrouter_fill_mix(mix: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BANDROUTER_FILL_MIX if mix is None else mix,
            0.0,
            1.0,
        )
    )


def normalize_live_fourphase_bandrouter_idle_floor(idle_floor: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BANDROUTER_IDLE_FLOOR if idle_floor is None else idle_floor,
            0.0,
            0.5,
        )
    )


def normalize_live_fourphase_bandrouter_pulse_interval_random_percent(value: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BANDROUTER_PULSE_INTERVAL_RANDOM_PERCENT if value is None else value,
            0.0,
            100.0,
        )
    )


def normalize_live_fourphase_bandrouter_post_projection_expansion(value: float | None) -> float:
    return float(
        np.clip(
            DEFAULT_LIVE_FOURPHASE_BANDROUTER_POST_PROJECTION_EXPANSION if value is None else value,
            0.0,
            2.0,
        )
    )


def normalize_live_fourphase_band_mapping(
    band_mapping: object | None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    default_mapping = DEFAULT_LIVE_FOURPHASE_BAND_MAPPING
    if not isinstance(band_mapping, (list, tuple)) or len(band_mapping) != 4:
        return default_mapping

    normalized_mapping: list[tuple[str, ...]] = []
    for index, default_bands in enumerate(default_mapping):
        raw_entry = band_mapping[index]
        if not isinstance(raw_entry, (list, tuple)):
            normalized_mapping.append(default_bands)
            continue

        selected: list[str] = []
        for raw_band in raw_entry:
            band_name = normalize_live_fourphase_band_name(raw_band)
            if band_name not in selected:
                selected.append(band_name)
            if len(selected) >= 3:
                break

        normalized_mapping.append(tuple(selected) if selected else default_bands)

    return tuple(normalized_mapping)  # type: ignore[return-value]


def normalize_live_fourphase_layout_model(layout_model: str | None) -> str:
    raw_value = str(layout_model or LIVE_FOURPHASE_STRAIGHT_LAYOUT).strip()
    if not raw_value:
        return LIVE_FOURPHASE_STRAIGHT_LAYOUT

    if raw_value.lower() in {"straight", LIVE_FOURPHASE_STRAIGHT_LAYOUT.lower()}:
        return LIVE_FOURPHASE_STRAIGHT_LAYOUT

    if raw_value in LAYOUT_MODELS:
        return raw_value

    if raw_value in LAYOUT_MODEL_ALIASES:
        return LAYOUT_MODEL_ALIASES[raw_value]

    lowered = raw_value.lower()
    for candidate in LIVE_FOURPHASE_LAYOUT_OPTIONS:
        if lowered == candidate.lower():
            return candidate
    for alias, resolved in LAYOUT_MODEL_ALIASES.items():
        if lowered == alias.lower():
            return resolved
    return LIVE_FOURPHASE_STRAIGHT_LAYOUT


def live_fourphase_layout_log_token(layout_model: str | None) -> str:
    resolved = normalize_live_fourphase_layout_model(layout_model)
    if resolved == LIVE_FOURPHASE_STRAIGHT_LAYOUT:
        return "straight"
    return resolved.lower().replace(" / ", "_").replace("/", "_").replace(" ", "_")


def live_fourphase_band_mapping_log_token(band_mapping: object | None) -> str:
    resolved_mapping = normalize_live_fourphase_band_mapping(band_mapping)
    return "/".join("+".join(group) for group in resolved_mapping)


def live_fourphase_beat_response_curve_log_token(curves: object | None) -> str:
    resolved_curves = normalize_live_fourphase_beat_response_curves(curves)
    return "/".join(resolved_curves)


def _apply_beat_fourphase_contrast(proximity: float, contrast_strength: float) -> float:
    clamped_proximity = float(np.clip(proximity, 0.0, 1.0))
    clamped_contrast = float(np.clip(contrast_strength, 0.0, 1.0))
    if clamped_contrast <= 1e-9:
        return clamped_proximity

    power = float(1.0 + clamped_contrast * 3.0)
    if clamped_proximity <= 0.5:
        normalized = float(np.clip(clamped_proximity * 2.0, 0.0, 1.0))
        return float(0.5 * np.power(normalized, power))

    normalized = float(np.clip(2.0 * (1.0 - clamped_proximity), 0.0, 1.0))
    return float(1.0 - 0.5 * np.power(normalized, power))


def _apply_beat_response_curve(proximity: float, response_curve: str) -> float:
    x = float(np.clip(proximity, 0.0, 1.0))
    curve = normalize_live_fourphase_beat_response_curve(response_curve)
    if curve == "ease":
        return float(np.clip(np.power(x, 1.8), 0.0, 1.0))
    if curve == "bell":
        if x <= 0.5:
            return float(2.0 * x * x)
        inv = float(1.0 - x)
        return float(1.0 - 2.0 * inv * inv)
    return x


def _resolve_beat_fourphase_contrast(
    *,
    radius_magnitude: float,
    orbit_angular_speed: float,
    radius_aware_contrast_strength: float,
    speed_threshold_spread_strength: float,
) -> float:
    radius_strength = normalize_live_fourphase_beat_radius_contrast_strength(
        radius_aware_contrast_strength
    )
    speed_strength = normalize_live_fourphase_beat_speed_spread_strength(
        speed_threshold_spread_strength
    )
    radius_normalized = float(np.clip(radius_magnitude, 0.0, 1.0))
    speed_threshold_rad_per_sec = 4.0
    speed_ceiling_rad_per_sec = 12.0
    speed_activation = float(
        np.clip(
            (abs(float(orbit_angular_speed)) - speed_threshold_rad_per_sec)
            / (speed_ceiling_rad_per_sec - speed_threshold_rad_per_sec),
            0.0,
            1.0,
        )
    )
    return float(np.clip(radius_strength * radius_normalized + speed_strength * speed_activation, 0.0, 1.0))


def _compute_classic_ring_levels(
    alpha: float,
    beta: float,
    *,
    beat_radius_contrast_strength: float | None = None,
    beat_speed_threshold_spread_strength: float | None = None,
    beat_response_curves: object | None = None,
    orbit_angular_speed: float | None = None,
) -> tuple[float, float, float, float]:
    x = float(-beta)
    y = float(alpha)
    radius = float(np.clip(np.hypot(x, y), 0.0, 1.0))
    theta = float(np.arctan2(y, x))

    contrast_strength = _resolve_beat_fourphase_contrast(
        radius_magnitude=radius,
        orbit_angular_speed=0.0 if orbit_angular_speed is None else orbit_angular_speed,
        radius_aware_contrast_strength=0.0 if beat_radius_contrast_strength is None else beat_radius_contrast_strength,
        speed_threshold_spread_strength=0.0 if beat_speed_threshold_spread_strength is None else beat_speed_threshold_spread_strength,
    )
    curves = normalize_live_fourphase_beat_response_curves(beat_response_curves)

    raw = np.asarray(
        [
            _apply_beat_response_curve(
                _apply_beat_fourphase_contrast((np.cos(theta - electrode_angle) + 1.0) * 0.5, contrast_strength),
                curves[index],
            )
            for index, electrode_angle in enumerate(_CLASSIC_RING_ELECTRODE_ANGLES)
        ],
        dtype=np.float64,
    )
    span = float(np.max(raw) - np.min(raw))
    if span <= 1e-12:
        scaled = np.zeros(4, dtype=np.float64)
    else:
        scaled = (raw - float(np.min(raw))) / span

    boundary = constrain_fourphase_coordinates(scaled[np.newaxis, :])[0]
    levels = _LIVE_FOURPHASE_CENTER + radius * (boundary - _LIVE_FOURPHASE_CENTER)
    return (
        float(levels[0]),
        float(levels[1]),
        float(levels[2]),
        float(levels[3]),
    )


def _expand_live_fourphase_levels(
    levels: tuple[float, float, float, float],
    expansion: float,
) -> tuple[float, float, float, float]:
    effective_expansion = float(np.clip(expansion, 0.0, 2.0))
    if abs(effective_expansion - 1.0) <= 1e-9:
        return levels

    level_array = np.asarray(levels, dtype=np.float64)
    expanded = _LIVE_FOURPHASE_CENTER + effective_expansion * (level_array - _LIVE_FOURPHASE_CENTER)
    constrained = constrain_fourphase_coordinates(expanded[np.newaxis, :])[0]
    clipped = np.clip(constrained, 0.0, 1.0)
    return (
        float(clipped[0]),
        float(clipped[1]),
        float(clipped[2]),
        float(clipped[3]),
    )


def _compute_tetra3d_levels(
    alpha: float,
    beta: float,
    sub_bass: float,
    *,
    layout_model: str = LIVE_FOURPHASE_STRAIGHT_LAYOUT,
    tetra_post_projection_expansion: float | None = None,
) -> tuple[float, float, float, float]:
    vertical = float(np.clip(sub_bass, 0.0, 1.0))
    xy_scale = float(np.clip(1.0 - 0.35 * vertical, 0.45, 1.0))
    projected = project_live_fourphase_point(
        alpha * xy_scale,
        beta * xy_scale,
        vertical,
        layout_model=layout_model,
        post_projection_expansion=tetra_post_projection_expansion,
    )
    return projected


def _compute_bandrouter_levels(
    *,
    band_levels: dict[str, float] | None,
    band_mapping: object | None,
    fill_angle: float,
    base: float,
    silence_fade: float,
    orbit_radius: float,
    bandrouter_fill_mix: float | None,
    bandrouter_idle_floor: float | None,
    bandrouter_post_projection_expansion: float | None,
) -> tuple[float, float, float, float]:
    normalized_mapping = normalize_live_fourphase_band_mapping(band_mapping)
    effective_fill_mix = normalize_live_fourphase_bandrouter_fill_mix(bandrouter_fill_mix)
    effective_idle_floor = normalize_live_fourphase_bandrouter_idle_floor(bandrouter_idle_floor)
    effective_post_projection_expansion = normalize_live_fourphase_bandrouter_post_projection_expansion(
        bandrouter_post_projection_expansion
    )
    resolved_levels = {band: 0.0 for band in LIVE_FOURPHASE_BAND_OPTIONS}
    if band_levels is not None:
        for band_name, value in band_levels.items():
            normalized_band = normalize_live_fourphase_band_name(band_name)
            resolved_levels[normalized_band] = float(np.clip(value, 0.0, 1.0))

    bloom_scale = float(0.85 + float(np.clip(orbit_radius, 0.35, 1.0)) * 0.20)
    dynamic_base = float(np.clip(base, 0.0, 1.0) * bloom_scale)
    dynamic_base = float(np.clip(dynamic_base, 0.0, 1.0))
    idle_level = float(dynamic_base * effective_idle_floor)
    fade = float(np.clip(silence_fade, 0.0, 1.0))

    raw_levels: list[float] = []
    for phase_index, bands in enumerate(normalized_mapping):
        spectral_sum = float(sum(resolved_levels[band] for band in bands))
        spectral_value = float(np.clip(spectral_sum, 0.0, 1.0))
        fill_proximity = float((np.cos(fill_angle - phase_index * (np.pi / 2.0)) + 1.0) * 0.5)
        level = dynamic_base * (
            spectral_value * _BANDROUTER_SPECTRAL_WEIGHT + fill_proximity * effective_fill_mix
        )
        raw_levels.append(float(np.clip(level * fade + idle_level, 0.0, 1.0)))

    raw_array = np.asarray(raw_levels, dtype=np.float64)
    activity_mix = float(np.clip(np.max(raw_array), 0.0, 1.0))
    constrained = constrain_fourphase_coordinates(raw_array[np.newaxis, :])[0]
    levels = _LIVE_FOURPHASE_CENTER + activity_mix * (constrained - _LIVE_FOURPHASE_CENTER)
    clipped = np.clip(levels, 0.0, 1.0)
    return _expand_live_fourphase_levels(
        (
            float(clipped[0]),
            float(clipped[1]),
            float(clipped[2]),
            float(clipped[3]),
        ),
        effective_post_projection_expansion,
    )


def compute_live_fourphase_bandrouter_pulse_interval_random_normalized(
    *,
    band_levels: dict[str, float] | None,
    pulse_interval_random_percent: float | None,
) -> float:
    resolved_levels = {band: 0.0 for band in LIVE_FOURPHASE_BAND_OPTIONS}
    if band_levels is not None:
        for band_name, value in band_levels.items():
            normalized_band = normalize_live_fourphase_band_name(band_name)
            resolved_levels[normalized_band] = float(np.clip(value, 0.0, 1.0))

    base_percent = normalize_live_fourphase_bandrouter_pulse_interval_random_percent(
        pulse_interval_random_percent
    )
    brilliance = float(np.clip(resolved_levels.get("brilliance", 0.0), 0.0, 1.0))
    dynamic_percent = base_percent + brilliance * (
        LIVE_FOURPHASE_BANDROUTER_MAX_BRILLIANCE_RANDOM_PERCENT - base_percent
    )
    return float(np.clip(dynamic_percent, 0.0, 100.0) / 100.0)


def project_live_fourphase_point(
    x: float,
    y: float,
    z: float,
    *,
    layout_model: str = LIVE_FOURPHASE_STRAIGHT_LAYOUT,
    post_projection_expansion: float | None = None,
) -> tuple[float, float, float, float]:
    resolved_layout = normalize_live_fourphase_layout_model(layout_model)
    if resolved_layout == LIVE_FOURPHASE_STRAIGHT_LAYOUT:
        projected = tetrahedral_project(
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            np.array([z], dtype=np.float64),
        )[0]
    else:
        projected = decode_layout_controls(
            np.array([z], dtype=np.float64),
            np.array([x], dtype=np.float64),
            np.array([y], dtype=np.float64),
            layout_model=resolved_layout,
        )[0]
    clipped = np.clip(projected, 0.0, 1.0)
    levels = (
        float(clipped[0]),
        float(clipped[1]),
        float(clipped[2]),
        float(clipped[3]),
    )
    return _expand_live_fourphase_levels(
        levels,
        normalize_live_fourphase_tetra_post_projection_expansion(post_projection_expansion),
    )


def compute_live_fourphase_levels(
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    *,
    model: str = "tetra3d",
    sub_bass: float = 0.0,
    layout_model: str = LIVE_FOURPHASE_STRAIGHT_LAYOUT,
    band_levels: dict[str, float] | None = None,
    band_mapping: object | None = None,
    fill_angle: float | None = None,
    base: float | None = None,
    silence_fade: float = 1.0,
    orbit_radius: float | None = None,
    bandrouter_fill_mix: float | None = None,
    bandrouter_idle_floor: float | None = None,
    tetra_post_projection_expansion: float | None = None,
    bandrouter_post_projection_expansion: float | None = None,
    beat_radius_contrast_strength: float | None = None,
    beat_speed_threshold_spread_strength: float | None = None,
    beat_response_curves: object | None = None,
    orbit_angular_speed: float | None = None,
) -> tuple[float, float, float, float]:
    """Map live alpha/beta(/gamma) motion to constrained direct E1-E4 levels."""
    normalized_model = normalize_live_fourphase_model(model)
    if normalized_model == "classic":
        return _compute_classic_ring_levels(
            alpha,
            beta,
            beat_radius_contrast_strength=beat_radius_contrast_strength,
            beat_speed_threshold_spread_strength=beat_speed_threshold_spread_strength,
            beat_response_curves=beat_response_curves,
            orbit_angular_speed=orbit_angular_speed,
        )

    if normalized_model == "bandrouter":
        effective_band_levels = dict(band_levels or {})
        if not effective_band_levels:
            effective_band_levels["sub_bass"] = float(np.clip(sub_bass, 0.0, 1.0))
        effective_fill_angle = float(np.arctan2(alpha, -beta) if fill_angle is None else fill_angle)
        effective_radius = float(
            np.clip(np.hypot(alpha, beta) if orbit_radius is None else orbit_radius, 0.0, 1.0)
        )
        effective_base = float(np.clip(effective_radius if base is None else base, 0.0, 1.0))
        return _compute_bandrouter_levels(
            band_levels=effective_band_levels,
            band_mapping=band_mapping,
            fill_angle=effective_fill_angle,
            base=effective_base,
            silence_fade=silence_fade,
            orbit_radius=effective_radius,
            bandrouter_fill_mix=bandrouter_fill_mix,
            bandrouter_idle_floor=bandrouter_idle_floor,
            bandrouter_post_projection_expansion=bandrouter_post_projection_expansion,
        )

    if abs(float(gamma)) > 1e-9:
        return project_live_fourphase_point(
            alpha,
            beta,
            gamma,
            layout_model=layout_model,
            post_projection_expansion=tetra_post_projection_expansion,
        )

    return _compute_tetra3d_levels(
        alpha,
        beta,
        sub_bass,
        layout_model=layout_model,
        tetra_post_projection_expansion=tetra_post_projection_expansion,
    )


def attach_cached_tcode_values(
    cmd: TCodeCommand,
    *,
    p0c0_enabled: bool,
    cached_p0_enabled: bool,
    cached_p0_val,
    cached_f0_enabled: bool,
    cached_f0_val,
    cached_p1_enabled: bool,
    cached_p1_val,
    cached_p3_enabled: bool,
    cached_p3_val,
    freq_window_ms: int,
) -> None:
    """Attach cached P0/C0/P1/P3 values to a command using existing send rules."""
    if p0c0_enabled and cached_p0_enabled and cached_p0_val is not None:
        cmd.pulse_freq = cached_p0_val

    if p0c0_enabled and cached_f0_enabled and cached_f0_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['C0'] = cached_f0_val

    if cached_p1_enabled and cached_p1_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P1'] = cached_p1_val
        cmd.tcode_tags['P1_duration'] = int(freq_window_ms)

    if cached_p3_enabled and cached_p3_val is not None:
        if cmd.tcode_tags is None:
            cmd.tcode_tags = {}
        cmd.tcode_tags['P3'] = cached_p3_val
        cmd.tcode_tags['P3_duration'] = int(freq_window_ms)


def apply_volume_ramp(
    cmd: TCodeCommand,
    *,
    volume_ramp_active: bool,
    volume_ramp_start_time: float,
    volume_ramp_duration: float,
    volume_ramp_from: float,
    volume_ramp_to: float,
    now: float | None = None,
) -> None:
    """Apply in-place volume ramp multiplier using existing linear ramp behavior."""
    if not volume_ramp_active:
        return

    current_time = time.time() if now is None else now
    elapsed = current_time - volume_ramp_start_time
    progress = min(1.0, elapsed / volume_ramp_duration)
    ramp_mult = volume_ramp_from + (volume_ramp_to - volume_ramp_from) * progress
    cmd.volume = cmd.volume * ramp_mult


def attach_direct_fourphase_levels(
    cmd: TCodeCommand,
    levels: tuple[float, float, float, float],
) -> None:
    if cmd.tcode_tags is None:
        cmd.tcode_tags = {}

    for index, value in enumerate(levels, start=1):
        tag = f"E{index}"
        tcode_value = int(max(0.0, min(1.0, float(value))) * 9999)
        cmd.tcode_tags[tag] = tcode_value
        cmd.tcode_tags[f"{tag}_duration"] = int(cmd.duration_ms)


def apply_live_output_mode(
    cmd: TCodeCommand,
    *,
    live_tcode_mode: str,
    live_fourphase_model: str = "tetra3d",
    live_fourphase_layout_model: str = LIVE_FOURPHASE_STRAIGHT_LAYOUT,
    live_fourphase_band_mapping: object | None = None,
    sub_bass: float = 0.0,
    band_levels: dict[str, float] | None = None,
    fill_angle: float | None = None,
    base: float | None = None,
    silence_fade: float = 1.0,
    orbit_radius: float | None = None,
    bandrouter_fill_mix: float | None = None,
    bandrouter_idle_floor: float | None = None,
    tetra_post_projection_expansion: float | None = None,
    bandrouter_post_projection_expansion: float | None = None,
    beat_radius_contrast_strength: float | None = None,
    beat_speed_threshold_spread_strength: float | None = None,
    beat_response_curves: object | None = None,
    orbit_angular_speed: float | None = None,
) -> None:
    """Adapt a live command to threephase (L0/L1) or fourphase (E1-E4) transport."""
    mode = str(live_tcode_mode or "threephase").strip().lower()
    if mode != "fourphase":
        cmd.include_linear_axes = True
        return

    cmd.include_linear_axes = False
    existing_tags = getattr(cmd, 'tcode_tags', {}) or {}
    if all(tag in existing_tags for tag in _DIRECT_FOURPHASE_TAGS):
        for tag in _DIRECT_FOURPHASE_TAGS:
            duration_key = f"{tag}_duration"
            existing_tags[duration_key] = int(existing_tags.get(duration_key, cmd.duration_ms) or cmd.duration_ms)
        cmd.tcode_tags = existing_tags
        return

    projected = compute_live_fourphase_levels(
        cmd.alpha,
        cmd.beta,
        0.0,
        model=live_fourphase_model,
        sub_bass=sub_bass,
        layout_model=live_fourphase_layout_model,
        band_levels=band_levels,
        band_mapping=live_fourphase_band_mapping,
        fill_angle=fill_angle,
        base=base,
        silence_fade=silence_fade,
        orbit_radius=orbit_radius,
        bandrouter_fill_mix=bandrouter_fill_mix,
        bandrouter_idle_floor=bandrouter_idle_floor,
        tetra_post_projection_expansion=tetra_post_projection_expansion,
        bandrouter_post_projection_expansion=bandrouter_post_projection_expansion,
        beat_radius_contrast_strength=beat_radius_contrast_strength,
        beat_speed_threshold_spread_strength=beat_speed_threshold_spread_strength,
        beat_response_curves=beat_response_curves,
        orbit_angular_speed=orbit_angular_speed,
    )
    attach_direct_fourphase_levels(cmd, projected)
