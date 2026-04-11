# PMV Funscript Generator — Implementation Guide

> **Date**: 2026-04-03  
> **Status**: Planning Revised — Repo Alignment Required Before Implementation  
> **Scope**: Standalone PyQt6 window for offline audio→multi-axis funscript generation

---

## Table of Contents

- [1. Overview](#1-overview)
- [2. Architecture](#2-architecture)
- [3. File Map](#3-file-map)
- [4. Dependencies](#4-dependencies)
- [5. Implementation Phases](#5-implementation-phases)
  - [Phase 1: Foundation & I/O](#phase-1-foundation--io)
  - [Phase 2: Audio Analysis Pipeline](#phase-2-audio-analysis-pipeline)
  - [Phase 3: Beat Detection Engine](#phase-3-beat-detection-engine)
  - [Phase 4: ML Intelligence Layer](#phase-4-ml-intelligence-layer)
  - [Phase 5: Position Mapping](#phase-5-position-mapping)
  - [Phase 6: Multi-Axis Conversion](#phase-6-multi-axis-conversion)
  - [Phase 7: Automap Optimization](#phase-7-automap-optimization)
  - [Phase 8: UI Controls Panel](#phase-8-ui-controls-panel)
  - [Phase 9: Visualization Panels](#phase-9-visualization-panels)
  - [Phase 10: Main Window & Pipeline Orchestration](#phase-10-main-window--pipeline-orchestration)
  - [Phase 11: Preset System & Config Compatibility](#phase-11-preset-system--config-compatibility)
  - [Phase 12: Integration & Polish](#phase-12-integration--polish)
- [6. Verification & Testing](#6-verification--testing)
- [7. Reference: Source Tool Comparison](#7-reference-source-tool-comparison)
- [8. Future Scope](#8-future-scope)

---

## 1. Overview

### What This Is

A **PMV (Porn Music Video) Funscript Generator** — a standalone tool that reads an audio file (or extracts audio from a video file) and produces one or more `.funscript` files mapping audio features to device motion. This is **not** a live/real-time feature; it processes the entire file offline, allowing multi-pass analysis, global normalization, and sophisticated ML inference.

### What It Combines

| Source | What We Take |
|--------|-------------|
| **bREadbeats** | 14-feature ML pipeline, multi-bus event detection, ACF tempo tracking, rule-fit model inference, P95 normalization |
| **PythonDancer** | Pitch+energy dual-component mapping, automap optimization (scipy Nelder-Mead), overflow modes (crop/bounce/fold), librosa beat/pitch extraction |
| **FunscriptGenerator v1.0** | FFT peak detection with lowpass/highpass filtering, peak/seek ratio control, stepped analysis workflow (Analyze → Generate) |
| **funscript-tools** | 1D→2D conversion (4 algorithms), E1-E4 response curves, auxiliary axes (frequency/volume/pulse), prostate algorithms, mixing ratios |

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Separate standalone window** | User preference; launchable from bREadbeats or independently |
| **Step-through workflow** (not one-click) | Like FunscriptGenerator: user clicks through discrete steps (Load → Analyze → Detect Beats → Generate → Export), inspecting results at each stage |
| **Audio extraction only** from video | Video motion analysis deferred to future scope |
| **Reuse by import, not duplication** | Reuse existing helpers and data models where APIs are real; build new offline feature extraction where the live code does not expose a reusable engine |
| **librosa for offline pitch/beat** | bREadbeats live pipeline lacks pitch extraction; librosa is the standard for offline audio analysis |
| **funscript-tools config import/export** | Users can share multi-axis presets between tools |
| **PyQt6-first UI integration** | The current application is already PyQt6, so PMV UI code must match the existing widget, signal, and threading patterns |
| **No hard dependency on third-party BPM services** | Offline BPM/key analysis must work locally; web metadata sources can only be optional enrichment |
| **All visualizations toggleable** | User selects which panels to show (any combination) |

### Implementation Readiness Update

This guide is now treated as a **repo-aligned implementation plan**, not a direct drop-in blueprint. Several assumptions from the initial draft do not match the current codebase:

1. **PyQt6, not PyQt5** — the existing application uses PyQt6 throughout, so all PMV UI code must target PyQt6 and reuse current helpers such as `SignalBridge`, `CollapsibleGroupBox`, and the existing slider widgets.
2. **Offline feature extraction must be built explicitly** — `audio_modules/feature_extractors.py` contains useful helper functions, but `FeatureExtractors.extract()` is not a usable offline engine in its current form. The PMV pipeline must compute the trained 14-feature schema directly.
3. **TempoTracker is not the offline BPM source** — offline BPM and phase need to come from librosa and/or a dedicated ACF pass first, then be adapted into `TempoState` for multi-bus corroboration.
4. **ML reuse means model/schema reuse, not full runtime-state reuse** — the safest path is to reuse the rule-fit JSON schema, feature ordering, normalization, and cadence thresholds while building an offline feature vector generator.
5. **Frozen builds need explicit work** — `main.spec` and `bREadbeats.spec` currently exclude `librosa`, `matplotlib`, and `scipy.optimize`, so packaging changes are part of the core plan, not late polish.

---

## 2. Architecture

### Pipeline Flow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         PMV FUNSCRIPT GENERATOR PIPELINE                        │
│                                                                                  │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────────┐  │
│  │  STEP 1  │    │  STEP 2   │    │  STEP 3   │    │  STEP 4   │    │   STEP 5   │  │
│  │  Load    │───▶│  Analyze  │───▶│  Detect   │───▶│  Generate │───▶│   Export    │  │
│  │  Audio   │    │  Features │    │  Beats    │    │  Script   │    │   Files    │  │
│  └─────────┘    └──────────┘    └──────────┘    └──────────┘    └────────────┘  │
│       │              │               │               │               │           │
│       ▼              ▼               ▼               ▼               ▼           │
│   Waveform      Feature         Beat markers    Position         .funscript     │
│   displayed     timeline        + tempo         timeline         files written   │
│                 computed        classified       + multi-axis                    │
│                                                  conversion                     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step-Through Workflow (Like FunscriptGenerator)

The user manually advances through each step, inspecting and tuning results before proceeding:

1. **Load Audio** — File picker or drag-drop. Audio waveform displayed immediately. If video, ffmpeg extracts audio first.
2. **Analyze** — Computes full feature timeline (bREadbeats 14-feature + pitch + energy). Progress bar per-substep. User sees spectral flux graph when done.
3. **Detect Beats** — Runs combined beat detection. User sees beat markers overlaid on waveform. Can adjust sensitivity and re-detect.
4. **Generate Script** — Maps features → positions → multi-axis conversion. User sees funscript timeline preview. Can adjust mapping params and re-generate.
5. **Export** — Writes selected axis .funscript files. Optionally exports CSV, heatmap.

Each step has its own "Run" button. The user can go back to any previous step without losing later results until they re-run that step.

### Module Dependency Graph

```
pmv_funscript_io.py ◄── (standalone, no deps)
         │
pmv_audio_analysis.py ◄── audio_modules/contracts.py
         │                  audio_modules/feature_extractors.py (helper funcs only)
         │                  audio_modules/volume_normalizer.py
         │                  frequency_utils.py
         │                  librosa, soundfile
         │
pmv_beat_engine.py ◄────── audio_modules/event_detector.py
         │                  audio_modules/tempo_tracker.py
         │                  audio_modules/contracts.py
         │                  librosa
         │
pmv_position_mapper.py ◄── beat_intelligence.py (model/schema reference)
         │                   datasets/rule_fit.json
         │                   audio_modules/contracts.py
         │
pmv_axis_converter.py ◄─── (standalone, numpy only)
         │
pmv_automap.py ◄────────── pmv_position_mapper.py
         │                   scipy.optimize
         │
pmv_controls.py ◄────────── widgets.py, PyQt6
         │
pmv_visualizations.py ◄──── pyqtgraph (primary)
         │                    matplotlib (optional heatmap only)
         │                    PyQt6
         │
pmv_generator.py ◄────────── ALL of the above
                              color_palette.py, stylesheet.py
                              widgets.py, threading
```

---

## 3. File Map

### New Files to Create (9 files)

| File | Purpose | Est. Lines |
|------|---------|-----------|
| `pmv_funscript_io.py` | Funscript/CSV read/write with metadata | ~150 |
| `pmv_audio_analysis.py` | Offline audio loading, OfflineFeatureExtractor wrapper, pitch/energy extraction | ~350 |
| `pmv_beat_engine.py` | Combined beat detection (librosa + multi-bus + FFT peaks) | ~400 |
| `pmv_position_mapper.py` | Feature→position mapping (PythonDancer + ML modulation) | ~300 |
| `pmv_axis_converter.py` | 1D→multi-axis (funscript-tools algorithms ported) | ~500 |
| `pmv_automap.py` | Scipy optimization for automatic parameter tuning | ~200 |
| `pmv_controls.py` | All control widgets in collapsible sections | ~600 |
| `pmv_visualizations.py` | Visualization panels (waveform, flux, heatmap, timeline, playback) | ~700 |
| `pmv_generator.py` | Main window, pipeline orchestration, step-through workflow | ~500 |

### Existing Files to Modify (4 files minimum)

| File | Change |
|------|--------|
| `main.py` | Add "PMV Funscript Generator" menu item to launch standalone window |
| `requirements.txt` | Add `librosa`, `soundfile` |
| `main.spec` | Remove packaging exclusions that would break PMV in frozen builds |
| `bREadbeats.spec` | Mirror the same packaging changes as `main.spec` |

### Existing Files to Import (no changes)

| File | What We Import |
|------|---------------|
| `audio_modules/feature_extractors.py` | Helper functions such as spectral flux, multiband energy, bass dominance |
| `audio_modules/contracts.py` | `FrontendFrame`, `FeatureFrame`, `BeatEvent`, `TempoState`, `TriggerDecision` |
| `audio_modules/event_detector.py` | `EventDetector`, `EventDetectorConfig` |
| `audio_modules/tempo_tracker.py` | `TempoTracker`, `TempoTrackerConfig` |
| `beat_intelligence.py` | Rule-fit loading conventions, feature ordering, cadence logic |
| `audio_modules/volume_normalizer.py` | P95 normalization utilities |
| `frequency_utils.py` | FFT helpers, band extraction |
| `color_palette.py` | Color constants |
| `stylesheet.py` | Qt stylesheet string |
| `config.py` | `Config`, `BeatDetectionConfig`, `AudioConfig` (for defaults) |
| `version.py` | `__version__` string for funscript metadata |

---

## 4. Dependencies

### New Python Packages

```
librosa>=0.10.0           # Offline audio analysis: beat tracking, pitch, PLP
soundfile>=0.12.0         # Audio file read/write (librosa backend)
```

### Already Present in requirements.txt

```
numpy                     # Array math
scipy                     # Nelder-Mead optimization (scipy.optimize.minimize)
PyQt6                     # GUI framework already used by the app
pyqtgraph                 # Preferred waveform/timeline rendering path for v1
```

### Optional Python Packages

```
matplotlib                # Optional: only if we keep a dedicated heatmap panel in v1
```

### External Tools

```
ffmpeg                    # Video→audio extraction (called via subprocess)
                          # Users must have ffmpeg on PATH or we bundle it
```

### Verification

After adding deps, run:
```powershell
pip install librosa soundfile
python -c "import librosa; import soundfile; print('OK')"
```

---

## 5. Implementation Phases

---

### Phase 1: Foundation & I/O

**File**: `pmv_funscript_io.py`  
**Dependencies**: None (standalone)  
**Est. Lines**: ~150

#### What to Build

Funscript file read/write module. This is dependency-free and can be built and tested immediately.

#### Detailed Specification

```python
# --- Data structures ---

@dataclass
class FunscriptAction:
    at: int           # Milliseconds from start
    pos: int          # Position 0-100

@dataclass
class FunscriptMetadata:
    creator: str = "bREadbeats PMV Generator"
    title: str = ""
    duration: int = 0        # ms
    description: str = ""
    performers: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    type: str = "basic"
    # PMV-specific metadata (stored under "pmv_params" key)
    parameters: dict = field(default_factory=dict)

# --- Functions ---

def write_funscript(
    path: str | Path,
    actions: list[FunscriptAction],
    metadata: FunscriptMetadata | None = None,
    inverted: bool = False,
    range_: int = 100
) -> None:
    """Write Funscript 1.0 JSON format."""

def read_funscript(path: str | Path) -> tuple[list[FunscriptAction], FunscriptMetadata]:
    """Read funscript file, return actions + metadata."""

def write_csv(path: str | Path, actions: list[FunscriptAction]) -> None:
    """Export as CSV: at_ms,position."""

def read_csv(path: str | Path) -> list[FunscriptAction]:
    """Import from CSV format."""

def actions_to_dict_list(actions: list[FunscriptAction]) -> list[dict]:
    """Convert to [{"at": ms, "pos": 0-100}, ...]."""

def dict_list_to_actions(data: list[dict]) -> list[FunscriptAction]:
    """Convert from raw dict list."""
```

#### Funscript 1.0 JSON Format

```json
{
    "version": "1.0",
    "inverted": false,
    "range": 100,
    "actions": [
        {"at": 0, "pos": 50},
        {"at": 500, "pos": 95},
        {"at": 1000, "pos": 5}
    ],
    "metadata": {
        "creator": "bREadbeats PMV Generator",
        "title": "Example",
        "duration": 180000,
        "description": "",
        "performers": [],
        "tags": [],
        "type": "basic"
    },
    "pmv_params": {
        "beat_sensitivity": 0.5,
        "pitch_range": 100,
        "energy_multiplier": 10,
        "ml_enabled": true,
        "ml_strength": 0.55,
        "axis_algorithm": "circular"
    }
}
```

#### Checkpoint ✅

- [ ] `write_funscript()` produces valid JSON openable in OFS (OpenFunscripter)
- [ ] `read_funscript()` successfully reads funscripts from `scripts/` directory
- [ ] Round-trip: write → read → write produces identical output
- [ ] CSV export/import works

#### Test

```python
# tests/test_pmv_funscript_io.py
def test_round_trip():
    actions = [FunscriptAction(0, 50), FunscriptAction(500, 95), FunscriptAction(1000, 5)]
    meta = FunscriptMetadata(title="Test", duration=1000)
    write_funscript(tmp / "test.funscript", actions, meta)
    read_actions, read_meta = read_funscript(tmp / "test.funscript")
    assert len(read_actions) == 3
    assert read_actions[0].at == 0
    assert read_actions[0].pos == 50
    assert read_meta.title == "Test"

def test_read_existing():
    actions, meta = read_funscript("scripts/CH-Tranquilizer.beta.funscript")
    assert len(actions) > 0
    assert all(0 <= a.pos <= 100 for a in actions)
```

---

### Phase 2: Audio Analysis Pipeline

**File**: `pmv_audio_analysis.py`  
**Dependencies**: `audio_modules/contracts.py`, `audio_modules/feature_extractors.py` (helper funcs), `audio_modules/volume_normalizer.py`, `frequency_utils.py`, `librosa`, `soundfile`  
**Est. Lines**: ~350

#### What to Build

Offline audio loading and full-file feature extraction. The core challenge is building an offline feature pipeline that matches the trained 14-feature schema used by bREadbeats without relying on the current live-only `FeatureExtractors.extract()` path.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class AnalysisConfig:
    sample_rate: int = 48000
    fft_size: int = 2048           # Configurable: 1024/2048/4096
    hop_size: int = 960            # 20ms at 48kHz
    window_size: int = 2208        # 46ms at 48kHz
    # Frequency filters (from FunscriptGenerator)
    lowpass_enabled: bool = False
    lowpass_hz: float = 1000.0
    highpass_enabled: bool = False
    highpass_hz: float = 400.0
    # Critical band range (from bREadbeats)
    freq_min_hz: float = 100.0
    freq_max_hz: float = 8000.0
    gain: float = 6.2

# --- Audio Timeline ---

@dataclass
class AudioTimeline:
    """Complete analysis results for one audio file."""
    samples: np.ndarray              # Raw mono audio samples
    sample_rate: int
    duration_ms: int
    # Per-frame data (hop-aligned)
    frame_times_ms: np.ndarray       # Time of each frame center
    feature_frames: list[FeatureFrame]  # bREadbeats 14-feature per frame
    # Per-frame extended features
    rms_per_frame: np.ndarray        # RMS energy per frame
    spectral_flux_per_frame: np.ndarray
    spectral_centroid_per_frame: np.ndarray
    spectral_flatness_per_frame: np.ndarray
    band_energies_per_frame: dict[str, np.ndarray]  # {bus_name: array}
    # 10-second rolling aggregates
    rms_mean_10s: np.ndarray
    rms_std_10s: np.ndarray
    flux_mean_10s: np.ndarray
    bass_mean_10s: np.ndarray
    energy_trend_10s: np.ndarray
    # Pitch (from librosa, per frame)
    pitch_per_frame: np.ndarray      # Log10-scaled fundamental frequency
    pitch_confidence: np.ndarray     # Magnitude-weighted confidence
    # Normalization stats (full-file P95)
    p95_flux: float
    p95_band_energies: dict[str, float]

# --- Functions ---

def load_audio(
    file_path: str | Path,
    config: AnalysisConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> np.ndarray:
    """
    Load audio file, extract from video if needed.
    Returns mono float32 samples at config.sample_rate.
    
    Steps:
    1. Check file extension. If video (.mp4, .mkv, .avi, .webm, .wmv, .mov),
       call ffmpeg to extract audio to temp WAV.
    2. Load via librosa.load(sr=config.sample_rate, mono=True)
    3. Return samples array
    """

def extract_video_audio(video_path: str | Path) -> Path:
    """
    Extract audio from video via ffmpeg subprocess.
    Returns path to temporary WAV file.
    
    Command: ffmpeg -i <video> -vn -acodec pcm_s16le -ar 48000 -ac 1 <temp.wav>
    Raises FileNotFoundError if ffmpeg not on PATH.
    """

def analyze_full_file(
    samples: np.ndarray,
    config: AnalysisConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> AudioTimeline:
    """
    Full-file feature extraction pipeline.
    
    Progress reports:
    - "Computing FFT frames..." (0-30%)
    - "Extracting bREadbeats features..." (30-50%)
    - "Extracting pitch..." (50-70%)
    - "Computing aggregates..." (70-85%)
    - "Normalizing..." (85-100%)
    """
```

#### Offline Feature Builder

The key design: **do not depend on `FeatureExtractors.extract()` as the offline engine**. Use the helper functions already present in `audio_modules/feature_extractors.py` where they are valid, but compute the per-frame feature payload explicitly.

```python
class OfflineFeatureExtractor:
    """
    Builds FeatureFrame-compatible data for offline analysis.
    
    Strategy:
    1. Compute STFT over entire file (numpy/scipy FFT)
    2. For each frame, construct a FrontendFrame with:
       - mono_time = frame_index * hop_size / sample_rate
       - wall_time = same (no real-time clock in offline mode)
       - spectrum = magnitude spectrum for this frame
       - band_energy = sum of critical-band magnitudes
       - spectral_flux = positive spectral difference from previous frame
       - raw_rms = RMS of time-domain windowed samples
       - raw_rms_db = 20*log10(raw_rms) or floor
    3. Compute the trained feature set directly:
       - band energies using the same frequency bounds expected by rule_fit.json
       - flux normalization, bass dominance, centroid, flatness, and deltas
       - 10-second aggregates matching the training feature order exactly
    4. Materialize FeatureFrame-compatible objects for downstream detector reuse
    """
    
    def __init__(self, config: AnalysisConfig):
        self._config = config
    
    def process_full_file(
        self,
        samples: np.ndarray,
        progress_callback: Callable[[float], None] | None = None
    ) -> list[FeatureFrame]:
        """Process entire audio array, return per-frame features."""
```

#### Pitch Extraction (from PythonDancer)

```python
def extract_pitch(
    samples: np.ndarray,
    sr: int,
    hop_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract pitch using librosa.piptrack().
    
    Returns:
        pitch_per_frame: Log10-scaled weighted mean pitch per frame
        confidence_per_frame: Magnitude-weighted confidence
    
    Algorithm (from PythonDancer):
    1. pitches, magnitudes = librosa.piptrack(y=samples, sr=sr, hop_length=hop_length)
    2. For each frame: weighted_pitch = sum(pitches * magnitudes) / sum(magnitudes)
    3. Log-scale: log10(weighted_pitch) where pitch > 0
    """
```

#### Frequency Filtering (from FunscriptGenerator)

```python
def apply_frequency_filters(
    spectrum: np.ndarray,
    freqs: np.ndarray,
    config: AnalysisConfig
) -> np.ndarray:
    """
    Apply lowpass/highpass filtering to FFT spectrum.
    Returns filtered magnitude spectrum.
    
    - If lowpass_enabled: zero out bins above lowpass_hz
    - If highpass_enabled: zero out bins below highpass_hz
    """
```

#### 10-Second Rolling Aggregates

```python
def compute_rolling_aggregates(
    feature_frames: list[FeatureFrame],
    frame_times_ms: np.ndarray,
    rms_per_frame: np.ndarray,
    flux_per_frame: np.ndarray,
    band_energies: dict[str, np.ndarray],
    window_sec: float = 10.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute 10-second rolling statistics matching bREadbeats' 14-feature set.
    
    Returns: rms_mean_10s, rms_std_10s, flux_mean_10s, bass_mean_10s, energy_trend_10s
    
    Offline advantage: uses centered window (look-ahead + look-behind),
    unlike live which can only look behind.
    """
```

#### P95 Normalization (Full-File)

```python
def p95_normalize(
    values: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Normalize by P95 percentile of entire file.
    Returns: (normalized_values, p95_value)
    
    Offline advantage: computed over entire file, not streaming EMA.
    """
```

#### Checkpoint ✅

- [ ] `load_audio()` loads WAV, MP3, FLAC, OGG files
- [ ] `extract_video_audio()` extracts audio from MP4 (requires ffmpeg on PATH)
- [ ] `analyze_full_file()` produces `AudioTimeline` with correct number of frames (`duration_ms / hop_ms ± 1`)
- [ ] `OfflineFeatureExtractor` produces `FeatureFrame`-compatible objects with non-placeholder values for the 14-feature model and detector inputs
- [ ] Pitch extraction produces non-zero values for tonal audio
- [ ] P95 normalization produces values in [0, ~1.5] range (some exceeding 1.0 is expected)
- [ ] Progress callback fires with monotonically increasing percentages
- [ ] Full pipeline runs on a 3-minute track in <30 seconds
- [ ] Feature column order and normalization inputs match `datasets/rule_fit.json` exactly

#### Test

```python
# tests/test_pmv_audio_analysis.py
def test_load_wav():
    samples = load_audio("test_data/sine_440hz.wav", AnalysisConfig())
    assert len(samples) > 0
    assert samples.dtype == np.float32

def test_offline_features_match_shape():
    samples = load_audio("test_data/sine_440hz.wav", AnalysisConfig())
    timeline = analyze_full_file(samples, AnalysisConfig())
    expected_frames = len(samples) // 960  # hop_size
    assert abs(len(timeline.feature_frames) - expected_frames) <= 1
    assert len(timeline.pitch_per_frame) == len(timeline.feature_frames)

def test_p95_normalization():
    values = np.random.exponential(1.0, 1000)
    normed, p95 = p95_normalize(values)
    assert np.percentile(normed, 95) == pytest.approx(1.0, abs=0.05)
```

---

### Phase 3: Beat Detection Engine

**File**: `pmv_beat_engine.py`  
**Dependencies**: `pmv_audio_analysis.py`, `audio_modules/event_detector.py`, `audio_modules/tempo_tracker.py`, `librosa`  
**Est. Lines**: ~400

#### What to Build

Combined beat detection merging three independent detectors with confidence scoring, deduplication, and beat classification. For repo fit, offline BPM and beat phase are estimated first, then adapted into `TempoState` for the multi-bus detector.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class BeatDetectionConfig:
    # General
    sensitivity: float = 0.5          # 0-1, maps to threshold multiplier
    refractory_ms: float = 170.0      # Min time between beats
    # Detector selection
    use_librosa: bool = True
    use_multibus: bool = True
    use_fft_peaks: bool = True
    # librosa settings
    plp_enabled: bool = True          # Probabilistic Latency Model
    # FFT peak settings (from FunscriptGenerator)
    peak_seek_ratio: float = 1.0
    peak_beat_threshold: float = 0.5
    # Multi-bus settings (from bREadbeats EventDetector)
    multibus_config: EventDetectorConfig = field(default_factory=EventDetectorConfig)

# --- Beat candidate ---

@dataclass
class BeatCandidate:
    time_ms: float
    confidence: float              # 0-1 combined confidence
    source: str                    # "librosa", "multibus", "fft_peak"
    bus_scores: dict[str, float] = field(default_factory=dict)
    beat_type: str = "beat"        # "downbeat", "beat", "syncopation"

# --- Output ---

@dataclass
class BeatTimeline:
    beats: list[BeatCandidate]     # Sorted by time_ms
    tempo_bpm: float               # Global estimated BPM
    tempo_confidence: float        # ACF confidence for global tempo
    beat_period_ms: float          # 60000 / tempo_bpm
    time_signature: int = 4        # Beats per bar (for downbeat classification)

# --- Functions ---

def detect_beats(
    timeline: AudioTimeline,
    config: BeatDetectionConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> BeatTimeline:
    """
    Run all enabled detectors, merge, deduplicate, classify.
    
    Progress reports:
    - "Running librosa beat detection..." (0-30%)
    - "Running multi-bus detection..." (30-60%)
    - "Running FFT peak detection..." (60-80%)
    - "Merging and classifying..." (80-100%)
    """
```

#### Detector 1: Librosa (Primary)

```python
def _detect_librosa(
    samples: np.ndarray,
    sr: int,
    plp_enabled: bool
) -> list[BeatCandidate]:
    """
    librosa.beat.beat_track() with optional PLP enhancement.
    
    Algorithm:
    1. tempo, beat_frames = librosa.beat.beat_track(y=samples, sr=sr, units='frames')
    2. If plp_enabled:
       pulse = librosa.beat.plp(y=samples, sr=sr)
       # Refine beat_frames using PLP peaks
    3. beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    4. Return BeatCandidates with source="librosa", confidence from onset strength
    """
```

#### Detector 2: Multi-Bus (Secondary, from bREadbeats)

```python
def _detect_multibus(
    timeline: AudioTimeline,
    config: EventDetectorConfig,
    sensitivity: float
) -> list[BeatCandidate]:
    """
    Run bREadbeats EventDetector offline over feature timeline.
    
    Algorithm:
    1. Create EventDetector with config
     2. Estimate global BPM + beat phase from librosa and/or offline ACF before this pass
     3. Build TempoState values per frame from that offline tempo solution
    3. For each frame in timeline.feature_frames:
         - Optionally use TempoTracker helper methods for smoothing, but not for tempo discovery
       - Call detector.detect(features, tempo, now_mono=frame_time)
       - If result.is_beat_candidate and beat_score > threshold:
         → Emit BeatCandidate with bus_scores
    4. Return candidates with source="multibus"
    """
```

#### Detector 3: FFT Peak Detection (Tertiary, from FunscriptGenerator)

```python
def _detect_fft_peaks(
    timeline: AudioTimeline,
    peak_seek_ratio: float,
    peak_beat_threshold: float,
    sensitivity: float
) -> list[BeatCandidate]:
    """
    FFT-based peak detection inspired by FunscriptGenerator.
    
    Algorithm:
    1. Compute spectral flux from timeline (already available)
    2. Apply peak_seek_ratio as novelty function multiplier
    3. Find local maxima in spectral flux where value > peak_beat_threshold
    4. Score each peak by relative height above local mean
    5. Return BeatCandidates with source="fft_peak"
    """
```

#### Merge & Deduplication

```python
def _merge_candidates(
    candidates: list[list[BeatCandidate]],
    refractory_ms: float
) -> list[BeatCandidate]:
    """
    Merge candidates from all detectors.
    
    Algorithm:
    1. Concatenate all candidate lists
    2. Sort by time_ms
    3. For candidates within refractory_ms of each other:
       - Keep highest-confidence candidate
       - If from different sources: boost confidence by 0.15 per corroborating source
       - Merge bus_scores from all corroborating candidates
    4. Return deduplicated list
    """
```

#### Beat Classification

```python
def _classify_beats(
    beats: list[BeatCandidate],
    tempo_bpm: float,
    time_signature: int = 4
) -> list[BeatCandidate]:
    """
    Classify each beat as downbeat, beat, or syncopation.
    
    Algorithm:
    1. beat_period_ms = 60000 / tempo_bpm
    2. Establish phase from first strong beat
    3. For each beat:
       - Compute expected beat grid position
       - phase_error = |actual_time - nearest_grid_point|
       - If phase_error < 25% of beat_period: on-grid
         - If beat_index % time_signature == 0: "downbeat"
         - Else: "beat"
       - If phase_error >= 25%: "syncopation"
    """
```

#### Global Tempo Estimation

```python
def _estimate_tempo(
    beats: list[BeatCandidate],
    timeline: AudioTimeline
) -> tuple[float, float]:
    """
    Estimate global BPM from beat timeline.
    
    Algorithm:
    1. Use librosa tempo estimate as the primary anchor for v1
    2. Optionally refine with offline ACF / IBI histogram periodicity
    3. Apply octave correction: check half/double BPM candidates
    4. Return (bpm, confidence)
    
    TempoTracker is treated as a helper for runtime-compatible tempo state, not the source of offline BPM discovery.
    """
```

#### Checkpoint ✅

- [ ] `detect_beats()` returns beats sorted by time_ms
- [ ] Beat count is reasonable for tempo (e.g., 120 BPM × 3 min = ~360 beats ±30%)
- [ ] Global tempo estimate within ±5% of librosa's estimate
- [ ] Downbeats appear every `time_signature` beats
- [ ] Syncopations have phase_error >= 25% of beat period
- [ ] Refractory deduplication eliminates sub-170ms duplicates
- [ ] Multi-source corroboration boosts confidence
- [ ] Progress callback fires correctly through each substep

#### Test

```python
# tests/test_pmv_beat_engine.py
def test_beat_count_reasonable():
    timeline = analyze_full_file(load_audio("test_data/120bpm.wav", cfg), cfg)
    result = detect_beats(timeline, BeatDetectionConfig())
    expected = 120 * (timeline.duration_ms / 60000)
    assert abs(len(result.beats) - expected) / expected < 0.3  # within 30%

def test_tempo_estimate():
    timeline = analyze_full_file(load_audio("test_data/120bpm.wav", cfg), cfg)
    result = detect_beats(timeline, BeatDetectionConfig())
    assert abs(result.tempo_bpm - 120) < 6  # within 5%

def test_deduplication():
    result = detect_beats(timeline, BeatDetectionConfig(refractory_ms=170))
    intervals = [b.time_ms - a.time_ms for a, b in zip(result.beats, result.beats[1:])]
    assert all(i >= 170 for i in intervals)
```

---

### Phase 4: ML Intelligence Layer

**File**: `pmv_position_mapper.py` (intelligence section)  
**Dependencies**: `beat_intelligence.py` (reference only), `audio_modules/contracts.py`, `pmv_audio_analysis.py`, `pmv_beat_engine.py`, `config.py`, `datasets/rule_fit.json`  
**Note**: This runs as part of the position mapper but is documented separately for clarity.

#### What to Build

Offline ML inference that predicts `speed_mult` and `cadence_hint` for each beat using the bREadbeats rule-fit model. The implementation should reuse the model schema, normalization, and cadence thresholds directly, rather than trying to reuse BeatIntelligence's live rolling-state machine.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class MLConfig:
    enabled: bool = True
    strength: float = 0.55           # 0-1, blend factor toward ML prediction
    cadence_mode: str = "auto"       # "auto", "fixed_1", "fixed_2", "fixed_4"
    rule_fit_path: str = ""          # Path to rule_fit.json (empty = use default)
    min_confidence: float = 0.12     # ACF confidence gate
    bidirectional_smooth: bool = True  # Look forward + backward (offline advantage)
    smooth_alpha: float = 0.15       # EMA smoothing factor

# --- Per-beat ML output ---

@dataclass
class BeatIntelligenceResult:
    speed_mult: float            # 0-1, predicted motion speed
    cadence_hint: int            # 1, 2, or 4 beats per stroke cycle
    energy_fullness: float       # 0-1, music intensity
    fill_gate_pass: bool         # Whether spectrum fill gate passed

# --- Functions ---

def compute_beat_intelligence(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    config: MLConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> list[BeatIntelligenceResult]:
    """
    Run ML inference for each beat.
    
    Algorithm:
     1. Load rule-fit model from JSON.
         Default resolution order:
         - custom `rule_fit_path` if provided
         - `config.beat.teaching_rule_fit_path`
         - fallback `datasets/rule_fit.json`
    2. For each beat:
       a. Extract 14-feature vector at beat's time from timeline
       b. Z-score normalize using model's stored mean/std
       c. Linear regression: speed_mult = intercept + sum(coef_i * feature_i)
       d. Apply cadence rule from model thresholds
    3. If bidirectional_smooth:
       Forward pass: EMA(alpha=smooth_alpha) over speed_mult
       Backward pass: EMA on reversed list
       Final = average of forward + backward
    4. Return per-beat results
    """
```

#### Feature Vector Construction

Matches bREadbeats' 14-feature FEATURE_COLUMNS:
```python
FEATURE_COLUMNS = [
    "rms",                  # raw_rms_db from nearest frame
    "spectral_flux",        # P95-normalized flux at beat time
    "sub_bass_energy",      # P95-normalized sub_bass band
    "low_mid_energy",       # P95-normalized low_mid band
    "mid_energy",           # P95-normalized mid band
    "high_energy",          # P95-normalized high band
    "low_high_ratio",       # (sub_bass + low_mid) / (high + eps)
    "spectral_centroid_hz", # Hz
    "spectral_flatness",    # 0-1
    "rms_mean_10s",         # 10s rolling mean of RMS
    "rms_std_10s",          # 10s rolling std of RMS
    "flux_mean_10s",        # 10s rolling mean of flux
    "bass_mean_10s",        # 10s rolling mean of sub_bass
    "energy_trend_10s",     # Linear slope over 10s window
]
```

#### Cadence Derivation

```python
def _derive_cadence(speed_mult: float, model: dict) -> int:
    """
    Map speed_mult to cadence using model's threshold rules.
    
    From datasets/rule_fit.json:
    - quiet_threshold: 0.300  → cadence=4 (slow, every 4th beat)
    - mid_threshold: 0.474    → cadence=2 (medium, every 2nd beat)
    - loud: speed > mid       → cadence=1 (every beat)
    """
```

#### Spectrum Fill Gate (Offline)

```python
def _spectrum_fill_gate(
    timeline: AudioTimeline,
    beat: BeatCandidate,
    beat_type: str
) -> bool:
    """
    Check whether the music at beat time is 'full enough' to trigger.
    
    Offline advantage: thresholds based on full-file percentile statistics
    rather than streaming EMA.
    
    Uses per-beat-type dBFS thresholds from bREadbeats:
    - Downbeat: -35 dBFS (easiest)
    - Beat: -40 dBFS  
    - Syncopation: -45 dBFS (hardest)
    """
```

#### Checkpoint ✅

- [ ] Rule-fit model loads from custom path, `config.beat.teaching_rule_fit_path`, or `datasets/rule_fit.json`
- [ ] 14-feature vector matches FEATURE_COLUMNS order
- [ ] `speed_mult` output in [0, 1] range
- [ ] `cadence_hint` is one of {1, 2, 4}
- [ ] Bidirectional smoothing reduces frame-to-frame jitter by >50%
- [ ] With ML disabled, function returns neutral defaults (speed_mult=0.5, cadence_hint=1)
- [ ] Custom model path works when specified
- [ ] Feature normalization uses the exact stored mean/std from the selected rule-fit JSON

#### Test

```python
# tests/test_pmv_ml_intelligence.py
def test_feature_vector_14_columns():
    features = build_feature_vector(timeline, beat)
    assert len(features) == 14
    assert all(k in features for k in FEATURE_COLUMNS)

def test_speed_mult_range():
    results = compute_beat_intelligence(timeline, beats, MLConfig())
    assert all(0.0 <= r.speed_mult <= 1.0 for r in results)

def test_cadence_derivation():
    assert _derive_cadence(0.2, model) == 4   # quiet
    assert _derive_cadence(0.4, model) == 2   # mid
    assert _derive_cadence(0.6, model) == 1   # loud

def test_ml_disabled_returns_defaults():
    results = compute_beat_intelligence(timeline, beats, MLConfig(enabled=False))
    assert all(r.speed_mult == 0.5 for r in results)
```

---

### Phase 5: Position Mapping

**File**: `pmv_position_mapper.py`  
**Dependencies**: `pmv_audio_analysis.py`, `pmv_beat_engine.py`, Phase 4 ML functions  
**Est. Lines**: ~300

#### What to Build

Map audio features + ML predictions to single-axis funscript positions. This is the core creative engine, combining PythonDancer's dual-component mapping with bREadbeats' ML modulation.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class MappingConfig:
    # Pitch mapping (from PythonDancer)
    pitch_range: float = 100.0       # -200 to 200
    amplitude_centering: float = 0.0  # -200 to 200
    center_offset: float = 0.0       # -300 to 300
    overflow_mode: str = "crop"      # "crop", "bounce", "fold"
    # Energy mapping (from PythonDancer)
    energy_multiplier: float = 10.0  # 0-100 (divided by 10 internally = 0-10)
    # ML modulation (from bREadbeats)
    ml_config: MLConfig = field(default_factory=MLConfig)
    # Timing
    min_command_delay_ms: float = 150.0  # Min time between actions
    points_per_second: int = 25          # Interpolation density
    # Min/max position
    pos_min: int = 0
    pos_max: int = 100

# --- Output ---

@dataclass
class PositionTimeline:
    actions: list[FunscriptAction]        # Final funscript actions
    beat_actions: list[FunscriptAction]   # Pre-interpolation (beat-aligned only)
    speed_profile: np.ndarray             # Speed at each action (for heatmap)
    ml_results: list[BeatIntelligenceResult] | None  # ML predictions per beat

# --- Functions ---

def generate_positions(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    config: MappingConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> PositionTimeline:
    """
    Full position mapping pipeline.
    
    Progress reports:
    - "Running ML intelligence..." (0-30%)
    - "Mapping positions..." (30-60%)
    - "Applying overflow..." (60-75%)
    - "Interpolating..." (75-90%)
    - "Computing speed profile..." (90-100%)
    """
```

#### Position Calculation (Core Algorithm)

```python
def _compute_raw_position(
    beat: BeatCandidate,
    timeline: AudioTimeline,
    ml_result: BeatIntelligenceResult | None,
    config: MappingConfig,
    is_upstroke: bool
) -> float:
    """
    Calculate raw position for one beat.
    
    Algorithm (PythonDancer + ML hybrid):
    
    1. Get normalized pitch and energy at beat time:
       pitch_norm = interpolate(timeline.pitch_per_frame, beat.time_ms)  # 0-1
       energy_norm = interpolate(timeline.rms_per_frame, beat.time_ms)  # 0-1
    
    2. Compute components:
       pitch_bias = (100 - config.pitch_range) / 2
       pitch_component = pitch_norm * config.pitch_range + pitch_bias
       energy_component = energy_norm * (config.energy_multiplier / 10) * 50
       centering_component = config.amplitude_centering * energy_norm
       
    3. Compute base offset:
       offset = pitch_component + centering_component + config.center_offset
       
    4. Compute raw position:
       position = energy_component + offset
       
    5. ML modulation (if enabled):
       ml_factor = 0.5 + 0.5 * ml_result.speed_mult * config.ml_config.strength
       position = position * ml_factor
       
    6. Direction:
       If is_upstroke: return position
       Else: return -position + offset  (downstroke mirrors upward)
    """
```

#### Cadence-Based Beat Filtering

```python
def _filter_beats_by_cadence(
    beats: list[BeatCandidate],
    ml_results: list[BeatIntelligenceResult],
    cadence_mode: str
) -> list[tuple[BeatCandidate, BeatIntelligenceResult]]:
    """
    Skip beats based on cadence.
    
    - cadence=1: every beat gets a stroke
    - cadence=2: every 2nd beat
    - cadence=4: every 4th beat
    
    If cadence_mode="auto": use ML's per-beat cadence_hint
    If cadence_mode="fixed_N": use fixed N for all beats
    
    Produces alternating up/down strokes for remaining beats.
    """
```

#### Overflow Handling (from PythonDancer)

```python
def _apply_overflow(
    actions: list[FunscriptAction],
    mode: str,
    pos_min: int = 0,
    pos_max: int = 100
) -> list[FunscriptAction]:
    """
    Handle positions outside [pos_min, pos_max].
    
    Modes:
    - "crop": Clamp to [pos_min, pos_max]
    - "bounce": Reflect off boundaries, inserting waypoints at boundary
    - "fold": Fold back into range (smoother than bounce)
    """
```

#### Interpolation

```python
def _interpolate_actions(
    actions: list[FunscriptAction],
    points_per_second: int,
    min_command_delay_ms: float
) -> list[FunscriptAction]:
    """
    Interpolate between beat-aligned actions for smoother motion.
    
    Algorithm:
    1. For each pair of consecutive actions:
       - Compute intermediate points at points_per_second rate
       - Linear interpolation between positions
    2. Enforce min_command_delay_ms between consecutive points
    3. Return densified action list
    """
```

#### Speed Profile

```python
def _compute_speed_profile(
    actions: list[FunscriptAction]
) -> np.ndarray:
    """
    Compute stroke speed at each action point (for heatmap visualization).
    speed[i] = |pos[i+1] - pos[i]| / (time[i+1] - time[i])  # units/ms
    """
```

#### Checkpoint ✅

- [ ] `generate_positions()` produces actions sorted by time
- [ ] All positions within [0, 100] after overflow handling
- [ ] Cadence=4 produces ~25% as many strokes as cadence=1
- [ ] Bounce overflow creates waypoints at boundaries (action count increases)
- [ ] Interpolation fills gaps with smooth transitions
- [ ] ML modulation changes amplitude (speed_mult=0 → small strokes, speed_mult=1 → large strokes)
- [ ] min_command_delay_ms respected between all adjacent actions
- [ ] Speed profile length matches action count

#### Test

```python
# tests/test_pmv_position_mapper.py
def test_positions_in_range():
    result = generate_positions(timeline, beats, MappingConfig())
    assert all(0 <= a.pos <= 100 for a in result.actions)

def test_cadence_reduces_beats():
    result_1 = generate_positions(timeline, beats, MappingConfig(ml_config=MLConfig(cadence_mode="fixed_1")))
    result_4 = generate_positions(timeline, beats, MappingConfig(ml_config=MLConfig(cadence_mode="fixed_4")))
    assert len(result_4.actions) < len(result_1.actions) * 0.5

def test_bounce_overflow():
    config = MappingConfig(pitch_range=200, overflow_mode="bounce")
    result = generate_positions(timeline, beats, config)
    assert all(0 <= a.pos <= 100 for a in result.actions)

def test_min_delay_enforced():
    result = generate_positions(timeline, beats, MappingConfig(min_command_delay_ms=150))
    for a, b in zip(result.actions, result.actions[1:]):
        assert b.at - a.at >= 150
```

---

### Phase 6: Multi-Axis Conversion

**File**: `pmv_axis_converter.py`  
**Dependencies**: `pmv_funscript_io.py` (FunscriptAction), numpy  
**Est. Lines**: ~500

#### What to Build

Convert single-axis funscript positions to multi-axis outputs. Ports algorithms from funscript-tools with full fidelity.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class AxisConfig:
    # 1D→2D conversion
    algorithm_2d: str = "circular"   # "circular", "top_left_right", "top_right_left", "360"
    min_distance: float = 0.1        # 0.1-0.9, minimum radius from center
    speed_threshold_pct: float = 50.0 # 0-100%, speed needed for max radius
    # Prostate
    prostate_algorithm: str = "standard"  # "standard", "tear_shaped"
    prostate_volume_mult: float = 1.5     # 1-3x intensity boost
    # E1-E4 curves
    e1_curve: str = "linear"            # "linear", "ease_in", "ease_out", "bell", "custom"
    e2_curve: str = "ease_in"
    e3_curve: str = "ease_out"
    e4_curve: str = "bell"
    e_custom_points: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    e_phase_shift: dict[str, float] = field(default_factory=lambda: {"e1": 0, "e2": 0, "e3": 0, "e4": 0})
    e_min_segment_sec: float = 0.5
    # Auxiliary mixing ratios (from funscript-tools)
    frequency_ramp_ratio: float = 2.0      # 1-10
    pulse_frequency_ratio: float = 3.0     # 1-10
    volume_ramp_ratio: float = 20.0        # 10-40
    pulse_rise_ratio: float = 2.0          # 1-10
    pulse_width_ratio: float = 3.0         # 1-10
    # Volume & rest
    rest_level: float = 0.4               # 0-1
    ramp_up_duration_sec: float = 1.0     # 0-10s
    ramp_pct_per_hour: float = 15.0       # 0-40%
    # Ranges
    pulse_freq_min: float = 0.40
    pulse_freq_max: float = 0.95
    pulse_rise_min: float = 0.00
    pulse_rise_max: float = 0.80
    pulse_width_min: float = 0.10
    pulse_width_max: float = 0.45
    # Speed calculation
    speed_window_sec: float = 5.0         # Rolling window for speed calc
    points_per_second: int = 25           # Output interpolation density
    # Output selection
    enabled_axes: set[str] = field(default_factory=lambda: {"main"})
    # Supported: "main", "alpha", "beta", "alpha_prostate", "beta_prostate",
    #            "e1", "e2", "e3", "e4", "frequency", "pulse_frequency",
    #            "volume", "pulse_rise", "pulse_width"

# --- Output ---

@dataclass
class MultiAxisResult:
    axes: dict[str, list[FunscriptAction]]  # axis_name → actions
    # Always contains "main" (the input single-axis)
```

#### 1D→2D Core Algorithm

```python
def convert_to_2d(
    main_actions: list[FunscriptAction],
    config: AxisConfig,
    duration_ms: int,
    progress_callback: Callable[[str, float], None] | None = None
) -> MultiAxisResult:
    """
    Master conversion function.
    
    Steps:
    1. Compute speed timeline from main_actions
    2. If alpha/beta enabled: run 2D conversion algorithm
    3. If prostate enabled: run prostate algorithm
    4. If E1-E4 enabled: run response curve mapping
    5. If auxiliary axes enabled: compute from derived signals
    6. Package all into MultiAxisResult
    """
```

#### Algorithm: Circular (0°-180°)

```python
def _convert_circular(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,           # 0-1 normalized speed at each action
    min_distance: float,
    speed_threshold_pct: float
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    """
    Semicircular 2D conversion.
    
    For each action:
    1. angle = position / 100 * π  (0° at pos=0, 180° at pos=100)
    2. radius = min_distance + (1.0 - min_distance) * (speed / threshold_speed)
       radius = clamp(radius, min_distance, 0.5)
    3. alpha = 0.5 + radius * cos(angle)   # center=0.5
       beta  = 0.5 + radius * sin(angle)   # center=0.5
    4. Scale alpha, beta to [0, 100] integer
    
    Returns: (alpha_actions, beta_actions)
    """
```

#### Algorithm: Top-Left-Right (0°-270°)

```python
def _convert_top_left_right(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    min_distance: float,
    speed_threshold_pct: float
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    """
    Counter-clockwise arc from top, spanning 270°.
    
    For each action:
    1. angle = position / 100 * (3π/2)  (0° at pos=0, 270° at pos=100)
    2. radius = speed-responsive (same formula as circular)
    3. alpha = 0.5 + radius * cos(angle + π/2)  # rotated to start at top
       beta  = 0.5 + radius * sin(angle + π/2)
    """
```

#### Algorithm: Top-Right-Left (0°-90°)

```python
def _convert_top_right_left(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    min_distance: float,
    speed_threshold_pct: float
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    """
    Clockwise mirrored arc from top, spanning 90°.
    
    For each action:
    1. angle = position / 100 * (π/2)
    2. radius = speed-responsive
    3. alpha = 0.5 + radius * cos(π/2 - angle)  # mirrored
       beta  = 0.5 + radius * sin(π/2 - angle)
    """
```

#### Algorithm: 360° (Restim Original)

```python
def _convert_360(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    min_distance: float
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    """
    Stroke-centered circular motion with random direction flips.
    
    For each stroke (min→max or max→min pair):
    1. center = midpoint of stroke
    2. radius = half of stroke length, speed adjusted
    3. direction = random choice of clockwise/counter-clockwise
    4. Map stroke progress to angle 0°→360° in chosen direction
    5. alpha = center + radius * cos(angle)
       beta  = center + radius * sin(angle)
    """
```

#### Speed Calculation

```python
def _compute_speed(
    actions: list[FunscriptAction],
    window_sec: float = 5.0
) -> np.ndarray:
    """
    Rolling-window speed computation.
    
    For each action:
    1. Look back window_sec seconds
    2. Sum position changes / sum time changes in window
    3. Normalize to [0, 1] by dividing by global max speed
    """
```

#### E1-E4 Response Curves

```python
PRESET_CURVES = {
    "linear":   [(0, 0), (1, 1)],
    "ease_in":  [(0, 0), (0.5, 0.2), (1, 1)],
    "ease_out": [(0, 0), (0.5, 0.8), (1, 1)],
    "bell":     [(0, 0), (0.25, 0.3), (0.5, 1.0), (0.75, 0.3), (1, 0)],
}

def _apply_response_curve(
    main_actions: list[FunscriptAction],
    curve_name: str,
    custom_points: list[tuple[float, float]] | None,
    phase_shift_pct: float = 0.0,
    min_segment_sec: float = 0.5
) -> list[FunscriptAction]:
    """
    Map main position through piecewise linear response curve.
    
    1. Normalize main pos to [0, 1]
    2. Interpolate through control points
    3. Apply phase shift (delay output by pct of segment duration)
    4. Scale back to [0, 100]
    """
```

#### Prostate: Tear-Shaped Algorithm

```python
def _convert_tear_shaped(
    main_actions: list[FunscriptAction],
    speed_norm: np.ndarray,
    min_distance: float,
    volume_mult: float
) -> tuple[list[FunscriptAction], list[FunscriptAction]]:
    """
    Asymmetric tear-drop pattern for prostate stimulation.
    
    For each local min/max pair:
    1. center = (midpoint_alpha=0.5, beta=0.5)
    2. Variable radius by angle:
       - 0°-120°: radius decreases linearly to min_distance
       - 120°-240°: constant min_distance (pointy end)
       - 240°-360°: radius increases back to full
    3. Full radius scaled by speed_norm * volume_mult
    """
```

#### Auxiliary Axes (funscript-tools mixing formula)

```python
def _mix(a: float, b: float, ratio: float) -> float:
    """
    Weighted mix: output = (a * (ratio - 1) + b) / ratio
    ratio=2 → 50/50, ratio=3 → 67/33, ratio=6 → 83/17
    """

def _generate_frequency(speed, ramp, ratio): ...
def _generate_pulse_frequency(speed, alpha, ratio): ...
def _generate_volume(ramp, speed, rest_level, ramp_up_sec, duration_ms): ...
def _generate_pulse_rise(inverted_signals, ratio): ...
def _generate_pulse_width(speed, inverted_main, ratio): ...
```

#### Checkpoint ✅

- [ ] Circular algorithm: alpha and beta both in [0, 100]
- [ ] With speed=0: radius = min_distance (small circle at center)
- [ ] With speed=max: radius reaches full range (0-100)
- [ ] All 4 algorithms produce distinct alpha/beta patterns
- [ ] E1-E4 response curves map correctly (bell curve peaks at 50% input → 100% output)
- [ ] Tear-shaped prostate has asymmetric radius profile
- [ ] Mixing formula: ratio=2 → equal blend, ratio=10 → 90% first input
- [ ] Output axis count matches `enabled_axes` selection
- [ ] Points per second interpolation produces smooth curves

#### Test

```python
# tests/test_pmv_axis_converter.py
def test_circular_center_at_rest():
    actions = [FunscriptAction(0, 50)]  # center position
    result = convert_to_2d(actions, AxisConfig(enabled_axes={"alpha", "beta"}), 1000)
    # At pos=50, angle=π/2, should be near top of semicircle
    assert 45 <= result.axes["alpha"][0].pos <= 55  # near center X
    assert result.axes["beta"][0].pos > 50  # above center Y

def test_speed_responsive_radius():
    # Fast strokes → larger radius, slow strokes → smaller radius
    fast = [FunscriptAction(0, 0), FunscriptAction(100, 100)]
    slow = [FunscriptAction(0, 45), FunscriptAction(200, 55)]
    fast_r = convert_to_2d(fast, cfg, 200)
    slow_r = convert_to_2d(slow, cfg, 400)
    fast_spread = max(a.pos for a in fast_r.axes["beta"]) - min(a.pos for a in fast_r.axes["beta"])
    slow_spread = max(a.pos for a in slow_r.axes["beta"]) - min(a.pos for a in slow_r.axes["beta"])
    assert fast_spread > slow_spread

def test_e_curves():
    actions = [FunscriptAction(i*100, int(i/10*100)) for i in range(11)]
    result = convert_to_2d(actions, AxisConfig(e1_curve="bell", enabled_axes={"e1"}), 1000)
    # Bell curve: input 50 should map to ~100
    mid_action = [a for a in result.axes["e1"] if 450 <= a.at <= 550][0]
    assert mid_action.pos > 80
```

---

### Phase 7: Automap Optimization

**File**: `pmv_automap.py`  
**Dependencies**: `pmv_position_mapper.py`, `scipy.optimize`  
**Est. Lines**: ~200

#### What to Build

Automatic parameter optimization via scipy Nelder-Mead. Port from PythonDancer with enhancements.

#### Detailed Specification

```python
# --- Configuration ---

@dataclass
class AutomapConfig:
    enabled: bool = False
    target_y_position: float = 20.0     # 0-100, target average action position
    target_speed: float = 250.0         # 0-400, target action speed
    target_speed_pct: float = 65.0      # 0-100, % of actions above target speed
    optimization_mode: str = "cmeanv2"  # "cmean", "cmeanv2", "clen"
    # Enhanced (beyond PythonDancer): also optimize ML strength
    optimize_ml_strength: bool = True

# --- Functions ---

def automap_optimize(
    timeline: AudioTimeline,
    beats: BeatTimeline,
    base_config: MappingConfig,
    automap_config: AutomapConfig,
    progress_callback: Callable[[str, float], None] | None = None
) -> MappingConfig:
    """
    Run Nelder-Mead optimization to find best mapping parameters.
    
    Returns: Optimized MappingConfig with tuned pitch_range, energy_multiplier,
             amplitude_centering, center_offset, and optionally ml_strength.
    
    Algorithm:
    1. Define objective function based on optimization_mode:
       - cmean: minimize variance of action speeds
       - cmeanv2: minimize |actual_pct_above_target - target_speed_pct|
       - clen: minimize |action_range - full_range|
    2. Define parameter vector:
       x = [pitch_range, energy_multiplier, amplitude_centering, center_offset]
       If optimize_ml_strength: x.append(ml_strength)
    3. Bounds from MappingConfig ranges
    4. scipy.optimize.minimize(method='Nelder-Mead', options={'xatol': 1e-10})
    5. Apply optimized parameters to copy of base_config
    6. Return optimized config
    
    Progress: report iteration count / estimated max iterations
    """
```

#### Objective Functions

```python
def _objective_cmean(x, timeline, beats, base_config):
    """Minimize speed variance (uniform speed distribution)."""
    config = _apply_params(base_config, x)
    result = generate_positions(timeline, beats, config)
    speeds = result.speed_profile
    return np.var(speeds)

def _objective_cmeanv2(x, timeline, beats, base_config, target_speed, target_pct):
    """Achieve target percentage of actions above target speed."""
    config = _apply_params(base_config, x)
    result = generate_positions(timeline, beats, config)
    speeds = result.speed_profile
    actual_pct = np.mean(speeds > target_speed / 400.0) * 100
    return abs(actual_pct - target_pct)

def _objective_clen(x, timeline, beats, base_config):
    """Optimize action range distribution (use full 0-100 range)."""
    config = _apply_params(base_config, x)
    result = generate_positions(timeline, beats, config)
    positions = [a.pos for a in result.actions]
    return -(max(positions) - min(positions))  # Negative: maximize range
```

#### Checkpoint ✅

- [ ] Optimization converges (objective function decreases over iterations)
- [ ] Optimized config produces positions closer to target Y than initial config
- [ ] cmeanv2 mode achieves within ±5% of target speed percentage
- [ ] ML strength optimization improves stroke quality score
- [ ] Runtime < 60 seconds for a typical 3-minute track (dozens of objective evaluations)
- [ ] Progress callback reports meaningful progress

#### Test

```python
# tests/test_pmv_automap.py
def test_optimization_improves_target():
    initial = generate_positions(timeline, beats, MappingConfig())
    initial_avg = np.mean([a.pos for a in initial.actions])
    
    optimized_config = automap_optimize(timeline, beats, MappingConfig(), AutomapConfig(target_y_position=50))
    optimized = generate_positions(timeline, beats, optimized_config)
    optimized_avg = np.mean([a.pos for a in optimized.actions])
    
    assert abs(optimized_avg - 50) < abs(initial_avg - 50)
```

---

### Phase 8: UI Controls Panel

**File**: `pmv_controls.py`  
**Dependencies**: PyQt6, `widgets.py` helpers  
**Est. Lines**: ~600

#### What to Build

All parameter controls organized in collapsible sections. Each section maps to a configuration dataclass from the pipeline modules.

Repo fit notes:
- Reuse `CollapsibleGroupBox`, `SliderWithLabel`, `RangeSliderWithLabel`, and `SignalBridge` patterns from the existing app.
- Do not introduce PyQt5-only widget or enum usage.
- Match the app's current threading model: worker functions in Python threads, GUI updates via Qt signals.

#### Widget Architecture

```python
class PMVControlsPanel(QScrollArea):
    """Main scrollable controls panel with collapsible sections."""
    
    # Signals emitted when any parameter changes
    config_changed = pyqtSignal()  # Generic change notification
    
    def get_analysis_config(self) -> AnalysisConfig: ...
    def get_beat_config(self) -> BeatDetectionConfig: ...
    def get_mapping_config(self) -> MappingConfig: ...
    def get_axis_config(self) -> AxisConfig: ...
    def get_automap_config(self) -> AutomapConfig: ...
    
    def set_from_preset(self, preset: dict) -> None: ...
    def to_preset(self) -> dict: ...
    
    # Import/export funscript-tools compatible config
    def import_funscript_tools_config(self, path: str) -> None: ...
    def export_funscript_tools_config(self, path: str) -> None: ...


class CollapsibleSection(QWidget):
    """Collapsible section with title bar that toggles content visibility."""
    def __init__(self, title: str, parent=None): ...
    def add_widget(self, widget: QWidget) -> None: ...
    def set_collapsed(self, collapsed: bool) -> None: ...
```

#### Section Layout

| Section | Controls | Source |
|---------|----------|--------|
| **Beat Detection** | Sensitivity (slider 0-1), Detection mode (combo), PLP toggle (checkbox), FFT length (combo 1024/2048/4096), Refractory ms (spinbox 50-500) | bREadbeats + PythonDancer |
| **Frequency Filters** | Lowpass enable + Hz (checkbox + spinbox), Highpass enable + Hz (checkbox + spinbox), EQ toggle | FunscriptGenerator |
| **Peak Detection** | Peak/Seek ratio (slider 0.1-5.0), Peak/Beat threshold (slider 0-1) | FunscriptGenerator |
| **Pitch Mapping** | Pitch range (slider -200 to 200), Amplitude centering (slider -200 to 200), Center offset (slider -300 to 300), Overflow mode (combo: Crop/Bounce/Fold) | PythonDancer |
| **Energy Mapping** | Energy multiplier (slider 0-100) | PythonDancer |
| **Timing** | Min command delay ms (spinbox 50-500), Points per second (spinbox 1-100) | FunscriptGenerator |
| **ML Intelligence** | Enabled toggle, Strength (slider 0-1), Cadence mode (combo: Auto/1/2/4), Model path (file picker) | bREadbeats |
| **Automap** | Enabled toggle, Target Y (spinbox 0-100), Target speed (spinbox 0-400), Target speed % (spinbox 0-100), Mode (combo: cmean/cmeanv2/clen) | PythonDancer |
| **Multi-Axis (2D)** | Algorithm (combo: Circular/TLR/TRL/360°), Min distance (slider 0.1-0.9), Speed threshold (slider 0-100%), Prostate algorithm (combo: Standard/Tear-Shaped), Prostate volume mult (slider 1-3) | funscript-tools |
| **E1-E4 Curves** | Per-axis curve selector (combo), Phase shift (slider 0-100%), Min segment duration (spinbox 0.1-5s) | funscript-tools |
| **Auxiliary Mixing** | Frequency ratio (slider 1-10), Pulse frequency ratio (slider 1-10), Volume ratio (slider 10-40), Rest level (slider 0-1), Ramp-up duration (spinbox 0-10s) | funscript-tools |
| **Output** | Axis checkboxes (main, alpha/beta, prostate, E1-E4, frequency, volume, pulse), Format (combo: Funscript/CSV) | All |

#### Step Buttons

Each processing step has its own button inside the controls panel (or at the top):

```python
class StepButtonBar(QWidget):
    """Row of step buttons showing pipeline progress."""
    
    # Buttons:
    # [1. Load Audio] → [2. Analyze] → [3. Detect Beats] → [4. Generate] → [5. Export]
    #     enabled         disabled         disabled            disabled        disabled
    #
    # After step 1 completes: step 2 becomes enabled, etc.
    # User can go back to any completed step and re-run it.
    
    step_requested = pyqtSignal(int)  # Emits step number (1-5)
    
    def set_step_enabled(self, step: int, enabled: bool) -> None: ...
    def set_step_status(self, step: int, status: str) -> None:
        """status: "ready", "running", "done", "error" """
```

#### Checkpoint ✅

- [ ] All sliders/spinboxes respect their documented min/max/default values
- [ ] `get_*_config()` methods produce valid config dataclass instances
- [ ] `config_changed` signal fires when any parameter changes
- [ ] Sections collapse/expand correctly
- [ ] `to_preset()` → `set_from_preset()` round-trips all values
- [ ] funscript-tools config import populates multi-axis section correctly
- [ ] Step buttons enable/disable in correct sequence

---

### Phase 9: Visualization Panels

**File**: `pmv_visualizations.py`  
**Dependencies**: PyQt6, pyqtgraph, optional matplotlib for heatmap only  
**Est. Lines**: ~700

#### What to Build

Five toggleable visualization panels sharing a synchronized time axis.

V1 preference: use `pyqtgraph` for waveform, flux, and position panels to stay aligned with the current app and minimize frozen-build complexity. Keep matplotlib optional for the heatmap only if pyqtgraph rendering is not sufficient.

#### Widget Architecture

```python
class VisualizationArea(QWidget):
    """Container for all visualization panels with toggle buttons."""
    
    def __init__(self, parent=None):
        # Toggle toolbar at top: [Waveform] [Flux] [Timeline] [Heatmap] [Playback]
        # Stacked panels below (visible/hidden based on toggles)
    
    def set_audio_data(self, samples: np.ndarray, sr: int) -> None: ...
    def set_features(self, timeline: AudioTimeline) -> None: ...
    def set_beats(self, beats: BeatTimeline) -> None: ...
    def set_positions(self, positions: PositionTimeline) -> None: ...
    def set_multi_axis(self, result: MultiAxisResult) -> None: ...
    def set_playback_position(self, time_ms: float) -> None: ...
    
    # Linked zoom/scroll across all visible panels
    def zoom_to_range(self, start_ms: float, end_ms: float) -> None: ...
```

#### Panel 1: Waveform + Beat Markers

```python
class WaveformPanel(QWidget):
    """Audio waveform with beat marker overlays."""
    
    # Display:
    # - Gray waveform envelope (amplitude over time)
    # - Vertical lines at beat positions:
    #   - Red = downbeat
    #   - Blue = beat
    #   - Green = syncopation
    # - Line opacity proportional to confidence
    # - Zoomable (mouse wheel), scrollable (click-drag)
    # - Time axis in MM:SS format
    
    def set_waveform(self, samples: np.ndarray, sr: int) -> None: ...
    def set_beats(self, beats: list[BeatCandidate]) -> None: ...
    def set_cursor(self, time_ms: float) -> None: ...
```

#### Panel 2: Spectral Flux Graph

```python
class SpectralFluxPanel(QWidget):
    """Spectral flux over time with threshold overlay."""
    
    # Display:
    # - Blue line: spectral flux per frame
    # - Orange dashed: detection threshold
    # - Yellow diamonds: detected onset peaks
    # - Zoomable/scrollable, linked to waveform panel
```

#### Panel 3: Funscript Position Timeline

```python
class PositionTimelinePanel(QWidget):
    """Generated positions plotted over time."""
    
    # Display:
    # - Y axis: position 0-100
    # - Main position: white/yellow line
    # - If multi-axis enabled:
    #   - Alpha: cyan line
    #   - Beta: magenta line
    #   - E1-E4: dimmer colored lines
    # - Beat markers at bottom (small ticks)
    # - Zoomable/scrollable, linked to other panels
    
    def set_positions(self, positions: PositionTimeline) -> None: ...
    def set_multi_axis(self, result: MultiAxisResult) -> None: ...
```

#### Panel 4: Speed Heatmap

```python
class SpeedHeatmapPanel(QWidget):
    """Speed heatmap visualization (from PythonDancer)."""
    
    # Display:
    # - Horizontal bar: time axis
    # - Color gradient: slow=blue → medium=green → fast=red
    # - Height represents speed magnitude
    # - Uses matplotlib colormaps for rendering
    
    def set_speed_profile(self, speed: np.ndarray, times_ms: np.ndarray) -> None: ...
```

#### Panel 5: Audio Playback with Sync

```python
class PlaybackPanel(QWidget):
    """Audio transport controls with visualization sync."""
    
    # Controls:
    # [Play/Pause] [Stop] [---|----seekbar----|---] [MM:SS / MM:SS] [Vol slider]
    #
    # Features:
    # - Play audio via soundfile + sounddevice (or QMediaPlayer)
    # - During playback, emit position updates at ~30Hz
    # - All other panels receive cursor position and draw vertical line
    # - Click anywhere on other panels to seek
    # - Mouse wheel on seekbar to zoom time axis
    
    position_changed = pyqtSignal(float)  # Emits current time in ms
    
    def load_audio(self, file_path: str) -> None: ...
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def seek(self, time_ms: float) -> None: ...
```

#### Synchronized Time Axis

```python
class TimeAxisSync:
    """Manages linked zoom/scroll across all visualization panels."""
    
    def __init__(self):
        self._panels: list[QWidget] = []
        self._view_start_ms: float = 0
        self._view_end_ms: float = 0
    
    def register_panel(self, panel: QWidget) -> None: ...
    def set_view_range(self, start_ms: float, end_ms: float) -> None:
        """All registered panels update to show this time range."""
    def zoom(self, center_ms: float, factor: float) -> None: ...
    def scroll(self, delta_ms: float) -> None: ...
```

#### Checkpoint ✅

- [ ] Waveform panel renders audio envelope correctly
- [ ] Beat markers appear at correct time positions with correct colors
- [ ] Spectral flux graph matches feature timeline data
- [ ] Position timeline correctly plots [0, 100] range
- [ ] Multi-axis overlay shows distinct colors per axis
- [ ] Speed heatmap color gradient is correct (blue=slow, red=fast)
- [ ] Audio playback cursor syncs across all visible panels
- [ ] Zoom/scroll affects all panels simultaneously
- [ ] Toggle buttons show/hide panels correctly
- [ ] Panel toggle state survives re-render

---

### Phase 10: Main Window & Pipeline Orchestration

**File**: `pmv_generator.py`  
**Dependencies**: All previous modules, PyQt6, `color_palette.py`, `stylesheet.py`, `widgets.py`, `threading`  
**Est. Lines**: ~500

#### What to Build

The main window that ties everything together. Implements the step-through workflow.

#### Window Layout

```
┌─────────────────────────────────────────────────────────────┐
│ PMV Funscript Generator                           [─][□][✕] │
├────────────────────┬────────────────────────────────────────┤
│                    │ ┌──────────────────────────────────┐   │
│  [Step Buttons]    │ │  Toggle: [Wave][Flux][Pos][Heat] │   │
│  ┌──────────────┐  │ ├──────────────────────────────────┤   │
│  │ 1. Load      │  │ │                                  │   │
│  │ 2. Analyze   │  │ │    Visualization Area            │   │
│  │ 3. Beats     │  │ │    (stacked panels)              │   │
│  │ 4. Generate  │  │ │                                  │   │
│  │ 5. Export    │  │ │                                  │   │
│  └──────────────┘  │ │                                  │   │
│                    │ │                                  │   │
│  ═══════════════   │ │                                  │   │
│  Controls Panel    │ │                                  │   │
│  (collapsible      │ ├──────────────────────────────────┤   │
│   sections,        │ │ [▶][■] ──|──────────── 01:23/03:45 │
│   scrollable)      │ │ Playback Transport               │   │
│                    │ └──────────────────────────────────┘   │
│                    │                                        │
│  ═══════════════   │  Status Bar: "Step 2/5: Analyzing..."  │
│  [Presets ▾]       │  [████████░░░░] 67%                    │
│  [Save Preset]     │                                        │
├────────────────────┴────────────────────────────────────────┤
│ File: C:\Music\song.mp3  │  BPM: 128  │  Duration: 3:45    │
└─────────────────────────────────────────────────────────────┘
```

#### Pipeline Orchestration

```python
class PMVGeneratorWindow(QMainWindow):
    def __init__(self, parent=None):
        # Layout setup
        # Wire step buttons to pipeline methods
        # Wire controls panel signals to re-run from affected step
    
    # --- Pipeline state ---
    _file_path: str | None
    _samples: np.ndarray | None
    _timeline: AudioTimeline | None
    _beats: BeatTimeline | None
    _positions: PositionTimeline | None
    _multi_axis: MultiAxisResult | None
    
    # --- Steps ---
    
    def step_1_load_audio(self):
        """
        1. Open file dialog (audio or video filter)
        2. If video: extract audio via ffmpeg (show progress)
        3. Load samples via load_audio()
        4. Display waveform immediately
        5. Enable Step 2 button
        6. Clear steps 2-5 results (if re-loading)
        """
    
    def step_2_analyze(self):
        """
        1. Read AnalysisConfig from controls panel
        2. Run analyze_full_file() in a background Python thread
        3. Progress bar updates via callback
        4. When done: populate spectral flux panel
        5. Enable Step 3 button
        6. Clear steps 3-5 results (if re-analyzing)
        """
    
    def step_3_detect_beats(self):
        """
        1. Read BeatDetectionConfig from controls panel
        2. Run detect_beats() in a background Python thread
        3. When done: overlay beat markers on waveform panel
        4. Display estimated BPM in status bar
        5. Enable Step 4 button
        6. Clear steps 4-5 results (if re-detecting)
        """
    
    def step_4_generate(self):
        """
        1. Read MappingConfig + AxisConfig + AutomapConfig from controls
        2. If automap enabled: run automap_optimize() first, update controls
        3. Run generate_positions() in a background Python thread
        4. If multi-axis enabled: run convert_to_2d()
        5. Populate position timeline + speed heatmap
        6. Enable Step 5 button
        """
    
    def step_5_export(self):
        """
        1. Open save dialog for output directory
        2. For each enabled axis in AxisConfig:
           write_funscript(dir / f"{stem}.{axis}.funscript", actions, metadata)
        3. If "main" in enabled_axes:
           write_funscript(dir / f"{stem}.funscript", main_actions, metadata)
        4. Optionally export CSV, heatmap PNG
        5. Show success message with file count
        """
```

#### Worker Signaling

```python
class PMVSignalBridge(QObject):
    """Thread-safe bridge for PMV pipeline updates."""
    progress = pyqtSignal(str, float)   # (message, 0-100)
    finished = pyqtSignal(object)       # Result object
    error = pyqtSignal(str)             # Error message


def run_pipeline_step(func, *args, signal_bridge: PMVSignalBridge, **kwargs) -> None:
    """Launch a daemon or managed Python thread, then emit Qt signals back to the UI."""
```

This matches the existing application's `threading` + Qt-signal approach more closely than introducing a separate QThread abstraction for each step.

#### Drag-and-Drop

```python
def dragEnterEvent(self, event: QDragEnterEvent):
    """Accept audio/video file drops."""
    if event.mimeData().hasUrls():
        url = event.mimeData().urls()[0].toLocalFile()
        if Path(url).suffix.lower() in SUPPORTED_EXTENSIONS:
            event.acceptProposedAction()

def dropEvent(self, event: QDropEvent):
    """Load dropped file."""
    url = event.mimeData().urls()[0].toLocalFile()
    self._file_path = url
    self.step_1_load_audio()
```

#### Supported File Extensions

```python
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.wma', '.m4a'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.webm', '.wmv', '.mov', '.flv'}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
```

#### Checkpoint ✅

- [ ] Window launches standalone (`python pmv_generator.py`)
- [ ] Window launches from bREadbeats menu item
- [ ] File open dialog filters for supported audio/video formats
- [ ] Drag-and-drop works for supported files
- [ ] Step buttons enable/disable in correct sequence
- [ ] Each step runs in background thread (UI stays responsive)
- [ ] Progress bar updates smoothly during each step
- [ ] Re-running a step clears downstream results
- [ ] Export creates correct number of .funscript files
- [ ] Status bar shows file info, BPM, duration
- [ ] Error in any step shows message box, doesn't crash

---

### Phase 11: Preset System & Config Compatibility

**Integrated into**: `pmv_controls.py` + `pmv_generator.py`

#### What to Build

Save/load parameter presets as JSON. Support import/export of funscript-tools config format.

#### Preset Format (Native)

```json
{
    "pmv_preset_version": 1,
    "name": "Balanced",
    "analysis": {
        "fft_size": 2048,
        "lowpass_enabled": false,
        "lowpass_hz": 1000,
        "highpass_enabled": false,
        "highpass_hz": 400
    },
    "beat_detection": {
        "sensitivity": 0.5,
        "use_librosa": true,
        "use_multibus": true,
        "use_fft_peaks": true,
        "plp_enabled": true,
        "peak_seek_ratio": 1.0,
        "peak_beat_threshold": 0.5,
        "refractory_ms": 170
    },
    "mapping": {
        "pitch_range": 100,
        "amplitude_centering": 0,
        "center_offset": 0,
        "overflow_mode": "crop",
        "energy_multiplier": 10,
        "min_command_delay_ms": 150,
        "points_per_second": 25
    },
    "ml": {
        "enabled": true,
        "strength": 0.55,
        "cadence_mode": "auto",
        "rule_fit_path": ""
    },
    "automap": {
        "enabled": false,
        "target_y": 20,
        "target_speed": 250,
        "target_speed_pct": 65,
        "mode": "cmeanv2"
    },
    "axis": {
        "algorithm_2d": "circular",
        "min_distance": 0.1,
        "speed_threshold_pct": 50,
        "prostate_algorithm": "standard",
        "prostate_volume_mult": 1.5,
        "e_curves": {"e1": "linear", "e2": "ease_in", "e3": "ease_out", "e4": "bell"},
        "frequency_ramp_ratio": 2,
        "pulse_frequency_ratio": 3,
        "volume_ramp_ratio": 20,
        "rest_level": 0.4,
        "ramp_up_duration_sec": 1
    },
    "output": {
        "enabled_axes": ["main"],
        "format": "funscript"
    }
}
```

#### Default Presets

| Preset | Description | Key Differences from Default |
|--------|-------------|------------------------------|
| **Balanced** | Good starting point for most music | All defaults |
| **High Energy** | Aggressive, fast strokes | sensitivity=0.7, energy_mult=20, cadence=1, pitch_range=150 |
| **Chill** | Slow, smooth motion | sensitivity=0.3, energy_mult=5, cadence=4, pitch_range=50 |
| **Beat Focused** | Emphasize beat detection, minimal pitch | pitch_range=0, ml_enabled=false, sensitivity=0.6 |
| **ML Driven** | Let ML model control everything | ml_strength=0.9, cadence=auto, sensitivity=0.4 |

#### Funscript-Tools Config Import/Export

```python
def import_funscript_tools_config(path: str) -> dict:
    """
    Read a funscript-tools config JSON and map to our AxisConfig fields.
    
    Mapping:
    - conversion_algorithm → algorithm_2d
    - min_distance_from_center → min_distance
    - speed_threshold_percent → speed_threshold_pct
    - prostate_algorithm → prostate_algorithm
    - prostate_volume_multiplier → prostate_volume_mult
    - frequency_ramp_combine_ratio → frequency_ramp_ratio
    - pulse_frequency_combine_ratio → pulse_frequency_ratio
    - volume_ramp_combine_ratio → volume_ramp_ratio
    - rest_level → rest_level
    - ramp_up_duration_after_rest → ramp_up_duration_sec
    - etc.
    """

def export_funscript_tools_config(axis_config: AxisConfig, path: str) -> None:
    """
    Write our AxisConfig as a funscript-tools compatible JSON.
    Only includes the multi-axis parameters that funscript-tools understands.
    """
```

#### Preset Storage

```
defaults/pmv_presets/
    balanced.json
    high_energy.json
    chill.json
    beat_focused.json
    ml_driven.json
user_pmv_presets/            # Created on first user save
    my_preset.json
```

#### Checkpoint ✅

- [ ] Save preset → creates valid JSON
- [ ] Load preset → all controls update to match
- [ ] Default presets all load without errors
- [ ] funscript-tools config import populates multi-axis controls
- [ ] funscript-tools config export creates file readable by funscript-tools
- [ ] Round-trip: native save → load preserves all values
- [ ] Missing fields in old presets handled gracefully (use defaults)

---

### Phase 12: Integration & Polish

#### 12a: bREadbeats Menu Integration

**File**: `main.py`  
**Change**: Add menu item or toolbar button

```python
# In the menu bar setup section of main.py:
pmv_action = QAction("PMV Funscript Generator", self)
pmv_action.triggered.connect(self._launch_pmv_generator)
# Add to appropriate menu (Tools or File menu)

def _launch_pmv_generator(self):
    from pmv_generator import PMVGeneratorWindow
    self._pmv_window = PMVGeneratorWindow(parent=None)
    self._pmv_window.show()
```

#### 12b: Requirements Update

**File**: `requirements.txt`

Add:
```
librosa>=0.10.0
soundfile>=0.12.0
```

If the heatmap panel keeps a matplotlib backend in v1, add `matplotlib` explicitly as well.

#### 12c: Frozen Build Integration

**Files**: `main.spec`, `bREadbeats.spec`

Both spec files must be updated before PMV can ship in packaged form.

Required changes:
- Remove `librosa` from `excludes`
- Remove `scipy.optimize` from `excludes`
- If matplotlib is retained, remove `matplotlib` from `excludes`
- Add any required hidden imports after validating the chosen implementation path
- Verify the frozen EXE can import all PMV dependencies on a clean Windows machine

#### 12d: Standalone Entry Point

`pmv_generator.py` should work as standalone:

```python
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    # Apply bREadbeats stylesheet if available
    try:
        from stylesheet import get_main_stylesheet
        app.setStyleSheet(get_main_stylesheet())
    except ImportError:
        pass
    window = PMVGeneratorWindow()
    window.show()
    sys.exit(app.exec_())
```

#### 12e: Preset Directory Creation

On first run, create `defaults/pmv_presets/` if missing and populate with defaults.

#### Checkpoint ✅

- [ ] `python pmv_generator.py` launches standalone window
- [ ] bREadbeats menu item launches PMV window
- [ ] `pip install -r requirements.txt` installs librosa and soundfile
- [ ] `main.spec` and `bREadbeats.spec` produce a frozen build that can import all PMV dependencies
- [ ] Default presets exist in `defaults/pmv_presets/`
- [ ] Window closes cleanly without affecting bREadbeats main window

---

## 6. Verification & Testing

### Packaging Validation

Add a frozen-build smoke test to the implementation plan.

- [ ] Source run: `python pmv_generator.py` works from the repo virtual environment
- [ ] Frozen run: packaged EXE launches PMV window without import errors
- [ ] If `ffmpeg` is missing, video extraction fails with a user-facing message rather than a raw traceback
- [ ] If matplotlib is retained, the heatmap panel renders correctly in the frozen build
- [ ] Clean Windows test machine confirms `librosa`, `soundfile`, and `scipy.optimize` are available in the packaged app

### Unit Tests

| Test File | Tests | Phase |
|-----------|-------|-------|
| `tests/test_pmv_funscript_io.py` | Round-trip write/read, read existing scripts, CSV export | Phase 1 |
| `tests/test_pmv_audio_analysis.py` | Audio loading, feature extraction shape, P95 normalization | Phase 2 |
| `tests/test_pmv_beat_engine.py` | Beat count, tempo estimate, dedup, classification | Phase 3 |
| `tests/test_pmv_ml_intelligence.py` | Feature vector, speed_mult range, cadence derivation | Phase 4 |
| `tests/test_pmv_position_mapper.py` | Position range, cadence filtering, overflow, min delay | Phase 5 |
| `tests/test_pmv_axis_converter.py` | Circular, TLR, TRL, 360°, E-curves, speed-radius, mixing | Phase 6 |
| `tests/test_pmv_automap.py` | Optimization convergence, target improvement | Phase 7 |

### Integration Tests

| Test | Description |
|------|-------------|
| **End-to-end single axis** | Load audio → analyze → detect → generate → export `.funscript` → verify valid JSON openable in OFS |
| **End-to-end multi-axis** | Same pipeline but with alpha/beta enabled → verify both `.alpha.funscript` and `.beta.funscript` created |
| **Preset round-trip** | Save preset → load preset → verify all pipeline outputs identical |
| **funscript-tools compat** | Export config → load in funscript-tools → verify parameters recognized |

### Manual Verification Checklist

| Check | Tool | What to Look For |
|-------|------|-----------------|
| Beat alignment | OFS (OpenFunscripter) | Load generated funscript alongside audio in OFS; beat markers should align with audible beats |
| Position range | OFS | Script should use reasonable range (not all stuck at 50, not all at 0/100) |
| Multi-axis motion | OFS 2D view | Alpha/beta should trace circular/arc pattern, not random noise |
| Speed heatmap | Built-in | Verify color matches perceived music intensity |
| Playback sync | Built-in | Play audio with cursor; cursor should track audio position accurately |

### Test Data Needed

| File | Purpose | Source |
|------|---------|--------|
| `test_data/sine_440hz.wav` | Basic audio loading test | Generate with scipy |
| `test_data/120bpm_click.wav` | Known-tempo beat detection test | Generate with known periodicity |
| `scripts/CH-Tranquilizer.beta.funscript` | Reference comparison | Already exists in repo |
| Any .mp4 file | Video audio extraction test | User-provided |

---

## 7. Reference: Source Tool Comparison

### Feature Matrix

| Feature | bREadbeats (live) | PythonDancer | FunscriptGen v1.0 | funscript-tools | **PMV Generator** |
|---------|:--:|:--:|:--:|:--:|:--:|
| Offline processing | ✗ | ✓ | ✓ | ✓ | **✓** |
| Real-time | ✓ | ✗ | ✗ | ✗ | ✗ |
| 14-feature ML | ✓ | ✗ | ✗ | ✗ | **✓** |
| Pitch extraction | ✗ | ✓ | ✗ | ✗ | **✓** |
| Beat detection (librosa) | ✗ | ✓ | ✗ | ✗ | **✓** |
| Beat detection (multi-bus) | ✓ | ✗ | ✗ | ✗ | **✓** |
| FFT peak detection | ✗ | ✗ | ✓ | ✗ | **✓** |
| Lowpass/highpass filters | ✗ | ✗ | ✓ | ✗ | **✓** |
| Automap optimization | ✗ | ✓ | ✗ | ✗ | **✓** |
| Multi-axis (alpha/beta) | ✓ (live) | ✗ | ✗ | ✓ | **✓** |
| E1-E4 response curves | ✗ | ✗ | ✗ | ✓ | **✓** |
| Prostate algorithms | ✗ | ✗ | ✗ | ✓ | **✓** |
| Auxiliary axes | ✗ | ✗ | ✗ | ✓ | **✓** |
| Overflow modes | ✗ | ✓ | ✗ | ✗ | **✓** |
| Cadence control (ML) | ✓ | ✗ | ✗ | ✗ | **✓** |
| Waveform visualization | ✗ | ✗ | ✓ | ✗ | **✓** |
| Speed heatmap | ✗ | ✓ | ✗ | ✗ | **✓** |
| Audio playback + sync | ✗ | ✗ | ✓ | ✗ | **✓** |
| Preset system | ✓ | ✗ | ✗ | ✓ | **✓** |
| funscript-tools config compat | ✗ | ✗ | ✗ | ✓ | **✓** |
| Step-through workflow | ✗ | ✗ | ✓ | ✗ | **✓** |
| Video audio extraction | ✗ | ✗ | ✓ | ✗ | **✓** |

### What We Gain vs. Each Tool

| vs. PythonDancer | What's Better |
|------------------|---------------|
| +ML intelligence | Speed/cadence adapts to music structure, not just energy |
| +Multi-bus detection | Parallel frequency-band analysis, not just librosa beat_track |
| +Multi-axis output | Alpha/beta/E1-E4/auxiliary axes (PythonDancer is single-axis only) |
| +Spectral flux | Better onset detection for non-tonal music |
| +Presets | Save and share parameter configurations |

| vs. FunscriptGenerator v1.0 | What's Better |
|------------------------------|---------------|
| +Pitch mapping | Position varies with musical pitch, not just timing |
| +ML modulation | Amplitude adapts to music intensity intelligently |
| +Multi-axis | Full axis suite instead of single-axis only |
| +Automap | Automatic parameter optimization |
| +Multiple beat detectors | librosa + multi-bus + FFT peaks combined |
| +More visualizations | Heatmap, flux graph, multi-axis overlay |

| vs. funscript-tools | What's Better |
|----------------------|---------------|
| +Audio-to-script generation | funscript-tools only converts existing scripts between axes |
| +ML intelligence | Audio-driven, not just geometry transforms |
| +Beat detection | Generates timing from scratch instead of requiring pre-made script |
| +Visualization + playback | Can preview and tune before exporting |

---

## 8. Future Scope

These items are **out of scope** for the initial implementation but documented for future work:

| Feature | Description | Priority |
|---------|-------------|----------|
| **Batch processing** | Process folder of audio/video files with same params | Medium |
| **Video motion analysis** | Extract visual motion from video frames for additional features | Medium |
| **Custom ML models** | Train per-genre or per-artist models from user-curated datasets | Low |
| **Real-time preview** | Play device motion in real-time during preview (connect to T-code) | Medium |
| **Multi-track** | Analyze multiple audio tracks (e.g., vocal vs. instrumental) separately | Low |
| **Funscript editor** | Manual post-processing of generated scripts within the tool | Low |
| **CLI batch mode** | `python pmv_generator.py --cli --input song.mp3 --preset balanced` | Medium |
| **Plugin system** | Allow custom position mappers as Python plugins | Low |

---

## Implementation Order Summary

```
Phase  1: pmv_funscript_io.py         ← Foundation, no deps, test immediately
Phase  2: pmv_audio_analysis.py       ← Core analysis, depends on bREadbeats modules
Phase  3: pmv_beat_engine.py          ← Beat detection, depends on Phase 2
Phase  4: ML intelligence (in mapper) ← Depends on Phase 2+3
Phase  5: pmv_position_mapper.py      ← Core generation, depends on Phase 2-4
Phase  6: pmv_axis_converter.py       ← Multi-axis, depends on Phase 1 (FunscriptAction)
Phase  7: pmv_automap.py              ← Optimization, depends on Phase 5
Phase  8: pmv_controls.py             ← UI controls, parallel with Phase 9
Phase  9: pmv_visualizations.py       ← Viz panels, parallel with Phase 8
Phase 10: pmv_generator.py            ← Main window, depends on everything
Phase 11: Presets + config compat     ← Integrated into Phase 8+10
Phase 12: Integration                 ← Final wiring into main.py
```

**Parallelization opportunities**:
- Phase 6 can start after Phase 1 (no dependency on Phase 2-5)
- Phase 8 and 9 can be built in parallel
- Phase 7 can start after Phase 5

**Total estimated new code**: ~3,700 lines across 9 files
