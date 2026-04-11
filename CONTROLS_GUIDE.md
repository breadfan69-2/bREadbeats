# bREadbeats PMV Generator — Controls Guide

A plain-English reference for every control in the PMV Generator window.
Controls are grouped the same way they appear in the UI.

---

## Pipeline Buttons (top bar)

| Button | What it does |
|--------|-------------|
| **1. Load Audio** | Opens a file picker to load your audio/video file. Everything starts here. |
| **2. Analyze** | Runs signal analysis on the loaded audio — extracts pitch, energy, spectral data, etc. Must be done before beat detection. |
| **3. Detect Beats** | Finds the beat timestamps in the analyzed audio. Produces the beat grid used for generation. |
| **4. Generate** | Converts beats + audio features into funscript positions using your current mapping settings. |
| **5. Export** | Saves the generated funscript(s) to disk. |

> Steps are enabled one at a time in order. You only need to re-run earlier steps if you change settings that affect them (e.g., changing Audio Analysis settings requires re-running Analyze and everything after it).

---

## Audio Analysis *(collapsed by default)*

These control how the audio file is decoded and what frequency content is extracted. **You rarely need to touch these.**

| Control | What it does |
|---------|-------------|
| **Sample Rate** | The number of audio samples per second used internally (default 48000 Hz). Higher = more accurate but slower. Stick to the default unless you know why you're changing it. |
| **FFT Size** | How many samples the frequency analyzer looks at in one chunk (1024 / 2048 / 4096). Larger = better low-frequency resolution but less time precision. Default 2048 is a good balance. |
| **Hop Size** | How many samples to advance between FFT windows. Smaller = more time-resolution frames per second. |
| **Window Size** | Size of the smoothing window applied around each FFT frame. |
| **Enable Lowpass** | Cuts out everything above **Lowpass Hz** before analysis. Useful if cymbals/highs are confusing the beat detector. |
| **Lowpass Hz** | The cutoff frequency for the lowpass filter (default 1000 Hz). Only active when Enable Lowpass is checked. |
| **Enable Highpass** | Cuts out everything below **Highpass Hz** before analysis. Useful if rumble/subs are dominating. |
| **Highpass Hz** | Cutoff frequency for the highpass filter (default 400 Hz). Only active when Enable Highpass is checked. |
| **Freq Min Hz** | The lowest frequency included in the pitch and energy calculations (default 100 Hz). |
| **Freq Max Hz** | The highest frequency included (default 8000 Hz). |
| **Gain** | Boosts the overall signal level before analysis (default 6.2). Raise if the audio is quiet and detections are sparse. |

---

## Beat Detection *(open by default)*

These control how beats are found in the analyzed audio. **This is the most important section for getting a good result.**

### Core Controls

| Control | What it does |
|---------|-------------|
| **Sensitivity** | Master knob for how easily a beat is triggered (0–1, default 0.5). Higher = more beats detected (may add false positives). Lower = only the clearest hits fire. |
| **Refractory (ms)** | Minimum gap between two detected beats (default 170 ms ≈ 350 BPM max). Prevents rapid double-triggers on a single hit. |

### Detector Switches

| Control | What it does |
|---------|-------------|
| **Use librosa detector** | Enables librosa's onset/beat tracker as one source of beats. Good for steady rhythmic music. |
| **Use multi-bus detector** | Enables the custom multi-frequency bus detector (see Bus Weights below). Best for bass-heavy or complex music. |
| **Use FFT peak detector** | Enables a simple spectral peak finder as a third beat source. |
| **Enable PLP enhancement** | Enables Predominant Local Pulse — a rhythm-tracking pass that helps line up beats with the musical grid. |

All enabled detectors vote together; the final beat list is merged from all of them.

### Peak Detector Controls

| Control | What it does |
|---------|-------------|
| **Peak/Seek Ratio** | Controls how far the peak detector searches around a candidate beat to find the true peak (default 1.0). Higher = broader search. |
| **Peak Beat Threshold** | Minimum score a candidate must reach to count as a beat in the peak detector (0–1, default 0.5). |

### Multi-Bus Detector Controls

The multi-bus detector splits audio into four frequency bands (sub-bass, low-mid, mid, high) and scores each frame on several criteria.

**Bus Weights** — These four sliders control how much each criterion contributes to the beat score. They should sum to roughly 1.0, but they don't have to be exact.

| Control | What it does |
|---------|-------------|
| **Bus Weight Flux** | Weight for spectral flux (how quickly energy is changing). High flux = fast transient = likely beat. (default 0.28) |
| **Bus Weight Band** | Weight for absolute band energy. Rewards high-energy moments. (default 0.30) |
| **Bus Weight Delta** | Weight for the rate of change between frames (energy delta). (default 0.17) |
| **Bus Weight Phase** | Weight for phase deviation — measures how much the signal's phase pattern changes, a hallmark of real transients. (default 0.20) |

**Bus Thresholds** — How the bus fires/releases on each beat candidate.

| Control | What it does |
|---------|-------------|
| **Bus Arm Threshold** | Score the bus must reach to "arm" (get ready to fire) (default 0.42). Lower = fires more easily. |
| **Bus Release Threshold** | Score must drop below this before the bus can re-arm (default 0.30). Prevents rapid re-triggering. |
| **Bus Refractory (ms)** | Minimum time between two bus-level fires, independent of the main refractory (default 170 ms). |

**Bus Classification**

| Control | What it does |
|---------|-------------|
| **Enable transient classification** | Classifies each beat as a transient (sharp attack) or sustain and adjusts scoring accordingly. Helps distinguish kick drums from long bass notes. |
| **Enable bass-dominance weighting** | Gives extra weight to beats where sub-bass / low-mid energy dominates. Good for music driven by kick/bass. |

---

## Pitch and Energy Mapping *(collapsed by default)*

These controls decide what **position** value each beat gets in the funscript — i.e., the stroke pattern.

| Control | What it does |
|---------|-------------|
| **Pitch Range** | How much the detected pitch drives the position output (−200 to 200, default 100). Positive = high pitch → high position. Zero = pitch has no effect. Negative = inverts the relationship. |
| **Amplitude Centering** | Shifts the center point of the position curve based on the audio's amplitude (−200 to 200, default 0). Positive values push the center point higher when the audio is louder. |
| **Center Offset** | A flat bias added to every position value (−300 to 300, default 0). Use this to shift the entire output up or down without changing the shape. Think of it as a "height" knob. |
| **Overflow Mode** | What happens when a calculated position goes outside [Position Min, Position Max]: **crop** = clamp to the boundary; **fold** = mirror back inward; **bounce** = elastic bounce. *(Bounce requires Min ≠ Max.)* |
| **Energy Multiplier** | Scales how much the audio's energy (loudness/intensity) affects position spread (0–100, default 10). Higher = louder sections create wider strokes. |
| **Min Command Delay (ms)** | The minimum time between two consecutive funscript commands (default 150 ms). Prevents commands from being issued faster than the device can actually complete them. |
| **Points per Second** | How many output data points per second are generated for the main axis (default 25). Higher = smoother but larger file. |
| **Position Min** | The lowest position value (0–100) the output is allowed to reach. Clamps the bottom of all strokes. |
| **Position Max** | The highest position value (0–100) the output is allowed to reach. Clamps the top of all strokes. |

---

## ML Intelligence *(collapsed by default)*

A rule-based ML layer that reads audio features and modulates the stroke pattern to make it feel more "musical" — varying cadence, speed, and fullness in sync with the track's energy patterns.

| Control | What it does |
|---------|-------------|
| **Enable ML modulation** | Turns the ML layer on or off entirely. Disabling it gives a "raw" mapping with no learned adjustments. |
| **ML Strength** | How strongly the ML layer overrides the base mapping (0–1, default 0.55). Higher = ML has more control. Lower = closer to the raw pitch/energy mapping. |
| **Cadence Mode** | How the ML decides stroke cadence (auto / fixed_1 / fixed_2 / fixed_4). **auto** = ML picks based on the music's rhythm. **fixed_1/2/4** = forces strokes in multiples of 1, 2, or 4 beats. |
| **Rule Fit Path** | File path to a custom trained rule-fit model (.json). Leave blank to use the built-in model. |
| **Teaching Rule Fit Path** | File path to a model trained from your personal Teaching Captures. Takes precedence over Rule Fit Path if both are set. |
| **Min Confidence** | The ML prediction confidence required before it overrides the base position (0–1, default 0.12). Lower = ML always applies even weak predictions. Higher = only high-confidence adjustments get through. |
| **Enable bidirectional smoothing** | Smooths the ML output both forward and backward in time, reducing jitter artifacts. |
| **Smooth Alpha** | The strength of the smoothing pass (0–1, default 0.15). Higher = heavier smoothing (less responsive to moment-to-moment changes). |

---

## Automap *(collapsed by default)*

Automap is an automatic optimizer. It runs the generator hundreds of times, searching for the best combination of **Pitch Range**, **Energy Multiplier**, **Amplitude Centering**, **Center Offset**, and optionally **ML Strength** to hit your target feel. It uses scipy's optimizer under the hood.

> **Note:** Automap is slow (seconds to minutes depending on Max Iterations). Live preview is paused while it is enabled.

| Control | What it does |
|---------|-------------|
| **Enable automap optimization** | Turns Automap on. When you click Generate, it runs the optimization before producing the final output. |
| **Target Y Position** | The desired average position value (0–100, default 20). The optimizer tries to make the mean stroke position match this. Lower = strokes centered toward the bottom. |
| **Target Speed** | The desired average stroke speed in units/sec (default 250). The optimizer adjusts parameters to approach this intensity. |
| **Target Speed Percent** | Percentage of strokes that should reach "full speed" (0–100, default 65). |
| **Optimization Mode** | The optimization objective function used: **cmean** = minimize distance from target Y mean; **cmeanv2** = refined mean+speed objective; **clen** = optimize stroke length distribution. |
| **Optimize ML Strength** | If checked, Automap also searches for the best ML Strength value, not just the mapping parameters. |
| **Max Iterations** | How many optimizer steps are allowed (default 120). More = potentially better result but much slower. |

---

## Multi Axis and Output *(collapsed by default)*

These controls generate additional **secondary axes** (alpha, beta, prostate, vibration, etc.) beyond the main stroke axis, and shape how the main position stream is converted into the multi-dimensional 2D output format.

### 2D Conversion

| Control | What it does |
|---------|-------------|
| **2D Algorithm** | How the main 1D position stream is turned into a 2D trajectory: **circular** = circular sweep (speed-scaled radius); **top_left_right** = alternates between top-left and top-right corners; **top_right_left** = same but mirrored; **360** = full-circle sweep. |
| **Min Distance** | Minimum radius of the 2D sweep pattern (0.1–0.5, default 0.1). Prevents the motion from collapsing to a single point during slow sections. |
| **Speed Threshold Percent** | The speed percentile at which the 2D sweep radius is at maximum (0–100, default 50). At speeds below this threshold, the radius scales down toward Min Distance. |

### Prostate Axis

| Control | What it does |
|---------|-------------|
| **Prostate Algorithm** | Shape of the prostate axis sweep: **standard** = linear in/out; **tear_shaped** = an asymmetric tear-drop curve that feels more natural. |
| **Prostate Volume Mult** | How much the audio volume scaling is amplified for the prostate axis (1–3, default 1.5). |

### E-Stim Axes (E1–E4)

Four independent e-stim (or vibration) output channels that follow the audio's energy contour.

| Control | What it does |
|---------|-------------|
| **E1–E4 Curve** | The intensity curve shape for each channel: **linear** = direct proportion; **ease_in** = starts slow, ramps up; **ease_out** = starts strong, eases off; **bell** = peaks in the middle. |
| **E1–E4 Phase Shift** | Offsets the timing of each channel's wave by a percentage of the current beat period (0–100%). Use this to stagger the channels so they don't all peak simultaneously. |
| **E Min Segment (s)** | Minimum duration of each e-stim output segment in seconds (default 0.5). Prevents tiny flickers. |

### Ramp Axes (Frequency / Volume / Pulse)

These outputs follow the audio's energy ramp — they respond to long-term tension/release curves, not individual beats.

| Control | What it does |
|---------|-------------|
| **Frequency Ramp Ratio** | How many times slower the frequency output changes compared to the raw energy signal (default 2.0). Higher = smoother, more gradual frequency changes. |
| **Pulse Frequency Ratio** | Same as above but for the pulse-frequency output channel (default 3.0). |
| **Volume Ramp Ratio** | Smoothing factor for the volume output channel (default 20.0). High by default because volume changes should feel gradual. |
| **Pulse Rise Ratio** | Smoothing factor for the pulse-rise output (default 2.0). |
| **Pulse Width Ratio** | Smoothing factor for the pulse-width output (default 3.0). |

### Timing

| Control | What it does |
|---------|-------------|
| **Rest Level** | The axis intensity during "silent" or low-energy sections (0–1, default 0.4). A value above 0 means the axis never fully stops. |
| **Ramp Up Duration (s)** | How many seconds after the start of the audio the output ramps from 0 up to full intensity (default 1.0). Prevents a jarring start. |
| **Speed Window (s)** | Time window (in seconds) used to calculate the local average speed for axis scaling (default 5.0). |
| **Axis Points per Second** | Resolution of all secondary axis outputs (default 25). Higher = smoother file, larger file size. |

### Enable Axes Checkboxes

| Axis | What it is |
|------|-----------|
| **main** | The primary up/down stroke axis. Should almost always be enabled. |
| **alpha** | Lateral left-right tilt derived from the 2D algorithm. |
| **beta** | Forward-back tilt derived from the 2D algorithm. |
| **alpha_prostate** | Lateral prostate axis. |
| **beta_prostate** | Forward-back prostate axis. |
| **e1 – e4** | Four independent e-stim/vibration channels. |
| **frequency** | Ramp-following vibration frequency channel. |
| **pulse_frequency** | Pulse modulation frequency channel. |
| **volume** | Volume/intensity channel. |
| **pulse_rise** | Controls how quickly each pulse rises. |
| **pulse_width** | Controls how wide (long) each pulse is. |

Only axes that are enabled will appear in the exported funscript file.

### Output

| Control | What it does |
|---------|-------------|
| **Output Format** | **funscript** = standard `.funscript` JSON (for any funscript-compatible player); **csv** = raw comma-separated values for analysis or custom tooling. |

---

## Quick-Start Recommended Settings

If you just want something that works, leave everything at defaults and only adjust these:

1. **Sensitivity** — If you get too few beats, raise it. Too many? Lower it.
2. **Pitch Range** — Start at 100. If strokes feel random/unintuitive relative to the music, try 50 or 0.
3. **Energy Multiplier** — If quiet sections have weak strokes and loud sections feel overwhelming, try 5–15.
4. **Center Offset** — If strokes feel too high or too low on average, nudge this.
5. **ML Strength** — Lower to 0.2–0.3 for more predictable patterns; raise toward 0.8 for more dynamic variation.
