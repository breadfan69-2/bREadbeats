# Session Notes — February 9, 2026

## Changes Made This Session

### 1. Auto-Freq Band Centers on Detected Peak (commit `43a7360`)
**Problem:** The auto-freq GUI band was stuck at ~half the spectrum width and centered on the middle of the entire range, not on the most consistent peak.

**Fix:** Rewrote the auto-freq section in `_update_display()` to:
- Always use spinbox width (`self._auto_freq_width`) as the band width, not an internal `_auto_freq_current_width` variable
- Always center the band on `_auto_freq_current_center` which tracks the detected peak via `find_consistent_frequency_band()`
- Removed the 3-phase scanning/closing/tracking state machine and gradual close-down — replaced with immediate spinbox-width display
- Center blends toward detected peak: 30% blend when scanning, 10% when locked

**Result:** -63 lines, +21 lines. Band immediately shows correct width centered on the best peak.

### 2. Gradual Widen on Beat Loss (pending commit)
**Problem:** When beats were lost, the band would oscillate/scroll around at minimum width before eventually expanding — visually jarring and not useful.

**Fix:** Replaced the snap-to-full-range behavior with smooth animated width:
- New `_auto_freq_current_width` variable tracks the *displayed* width (animated)
- When beats are arriving: width gradually narrows toward spinbox target (15% per 100ms update)
- When beats are missed: width gradually widens toward full range (10% per 100ms update)  
- Center position **freezes** during widening (no more scrolling/chasing new peaks while expanding)
- On 3 missed beats: tempo unlocks + energy history clears, but width widens smoothly instead of snapping
- New label state: `⚡[WIDEN]` shown during widening phase

**GUI label states:**
- `🔓[SCAN]` — scanning, tempo not locked
- `🔒[TRACK]` — tracking, tempo locked
- `⚡[WIDEN]` — beats missed, band expanding

---

## Peak Detection Research — Implementation Guide

### Source 1: Smoothed Z-Score Algorithm (Brakel, 2014)
**Source:** https://stackoverflow.com/a/22640362  
**Priority:** HIGH — most directly applicable

#### Algorithm
```python
# Parameters:
#   lag       = rolling window size (number of past samples)
#   threshold = number of std devs to trigger signal (e.g., 3.5)
#   influence = how much peaks affect the rolling stats (0=ignore, 1=full)

# For each new data point:
if abs(new_value - rolling_mean) > threshold * rolling_std:
    signal = +1 (peak) or -1 (valley)
    filtered_value = influence * new_value + (1 - influence) * prev_filtered
else:
    signal = 0
    filtered_value = new_value

rolling_mean = mean(filtered_values[i-lag:i])
rolling_std  = std(filtered_values[i-lag:i])
```

#### How to Integrate into bREadbeats

**Option A — Band Quality Scoring (replace CV scoring in `find_consistent_frequency_band`):**
1. For each candidate band (scanning in 50Hz steps across 30-22050Hz):
   - Maintain a z-score detector on that band's energy history
   - Count how many z-score signals (peaks) it produces over the last N samples
   - Score = signal_count / total_samples (regularity metric)
2. Band with highest regular signal rate = best for beat detection
3. Parameters: `lag=16` (half our 32-sample history), `threshold=2.5`, `influence=0`

**Option B — Self-Tuning Beat Detection (replace peak_floor + rise threshold):**
1. Feed `band_energy` values into a single z-score detector each frame
2. When `signal == +1`: that's a beat (energy spike above adaptive threshold)
3. The rolling mean/std automatically adapts to the audio level
4. `peak_floor` becomes unnecessary — the z-score threshold replaces it
5. Parameters: `lag=30` (~0.5s at 60fps), `threshold=3.0`, `influence=0.1`

**Streaming Python class available** — processes one point at a time, no recalculation of history. Would live in `audio_engine.py`.

```python
class ZScorePeakDetector:
    """Real-time z-score peak detector for streaming data."""
    def __init__(self, lag=30, threshold=3.0, influence=0.1):
        self.lag = lag
        self.threshold = threshold
        self.influence = influence
        self.buffer = []
        self.filtered = []
        self.mean = 0.0
        self.std = 0.0
        self.initialized = False
    
    def update(self, value: float) -> int:
        """Feed one value, returns +1 (peak), -1 (valley), or 0."""
        self.buffer.append(value)
        if len(self.buffer) < self.lag:
            self.filtered.append(value)
            return 0
        if not self.initialized:
            self.mean = np.mean(self.buffer[:self.lag])
            self.std = np.std(self.buffer[:self.lag])
            self.filtered.append(value)
            self.initialized = True
            return 0
        
        if abs(value - self.mean) > self.threshold * self.std:
            signal = 1 if value > self.mean else -1
            filt = self.influence * value + (1 - self.influence) * self.filtered[-1]
        else:
            signal = 0
            filt = value
        
        self.filtered.append(filt)
        # Update rolling stats from filtered buffer
        window = self.filtered[-self.lag:]
        self.mean = np.mean(window)
        self.std = np.std(window)
        
        # Keep memory bounded
        if len(self.buffer) > self.lag * 3:
            self.buffer = self.buffer[-self.lag * 2:]
            self.filtered = self.filtered[-self.lag * 2:]
        
        return signal
```

---

### Source 2: Derivative-Based Peak Detection (UMD)
**Source:** https://terpconnect.umd.edu/~toh/spectrum/PeakFindingandMeasurement.htm  
**Priority:** MEDIUM — useful for auto-tuning parameters

#### Key Concepts
- Smooth the first derivative, then find downward zero-crossings
- Two thresholds: **SlopeThreshold** (rejects broad/narrow features) and **AmpThreshold** (rejects small peaks)
- Optimal SlopeThreshold ≈ `0.7 × WidthPoints^-2`

#### How to Integrate
- **Auto-calculate `rise_sens`:** If we know the typical beat width in samples (from beat interval / sample rate), we can compute optimal rise_sens as `0.7 / width_samples²`
- **Segmented detection:** Use different detection parameters for different frequency ranges — low frequencies need wider smoothing windows, high frequencies need narrower ones
- Could improve `find_consistent_frequency_band()` by using derivative zero-crossings to count peaks per band instead of raw energy variance

---

### Source 3: NI Quadratic Fit Peak Detection
**Source:** https://www.ni.com/en/support/documentation/supplemental/06/peak-detection-using-labview-and-measurement-studio.html  
**Priority:** LOW — validates our current approach

#### Key Takeaways
- Fits parabola to groups of `width` points, checks concavity for peak/valley
- Returns **fractional index** locations (sub-sample accuracy)
- Multi-block processing with retained internal state between calls
- **Key advice:** "Smooth first, then detect with width=3" — validates our Butterworth-filter-first approach
- Interpolation before detection improves accuracy — we could interpolate FFT bins

#### Potential Use
- Not worth implementing separately — our Butterworth + threshold approach is already similar
- The "smooth first, detect with minimal width" advice confirms our architecture is sound

---

### Source 4: MathWorks Trough Detection
**Source:** https://www.mathworks.com/matlabcentral/answers/2042461
**Priority:** LOW — just validates derivative + zero-crossing approach

Confirms windowed smoothing + first derivative sign change is standard for real-time trough/peak detection in streaming signals. No novel techniques beyond what Sources 1-2 cover.

---

## Recommended Implementation Order

1. **Z-Score band scoring** (Option A above) — Drop-in replacement for `find_consistent_frequency_band()` scoring. Low risk, improves band selection quality.

2. **Z-Score beat detection** (Option B above) — More invasive change. Would replace `peak_floor` entirely with adaptive threshold. High reward but needs careful testing since it changes core beat detection.

3. **Auto-tuned rise_sens** — Use derivative-based formula to auto-calculate optimal rise sensitivity from detected beat width. Could integrate with existing auto-adjust hunting system.
