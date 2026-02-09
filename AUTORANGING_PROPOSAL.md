# Real-Time Metric-Based Auto-Ranging Proposal

## Current System Analysis

### Timer-Based Hunting Cycle (Current)
- **14-step sequence** with strategic flux_mult repetition
- **1/8 interval (12.5ms)** during HUNTING, normal interval during REVERSING
- **Per-parameter state machine** (HUNTING → REVERSING → LOCKED)
- **Limitations:**
  - Blind adjustments on a fixed schedule, not responsive to actual audio conditions
  - No feedback on whether parameters are actually helping or hurting
  - Random oscillation added to parameters (sine wave at 0.75× step_size)
  - Takes time to converge even when parameters are already good
  - Cycle runs regardless of how far audio is from optimal target zone

### Available Real-Time Metrics (Currently Unused)
From [audio_engine.py](audio_engine.py):

1. **band_energy** (line ~360) - Current energy level in detection band
2. **peak_envelope** - Running peak (decays at 0.85 per update)
3. **intensity** - Normalized: `band_energy / peak_envelope` (0-1)
4. **spectral_flux** - Change in spectrum between frames
5. **measure_energy_accum[4]** - Accumulated energy per beat position
6. **downbeat_confidence** - Energy ratio (strongest_pos / mean_energy)
7. **beat_stability** - Coefficient of variation of beat intervals
8. **smoothed_tempo** - Exponentially smoothed BPM

---

## Proposed Real-Time Metric-Based System

### Core Principle
**Stop running a timer cycle. Instead, measure margins and adjust continuously based on real-time feedback.**

Three independent feedback loops, each with its own metric:

### 1. Peak Floor Optimization

**Current Behavior:**
- Static threshold; user sets it manually
- Auto-adjust cycles through it in a sequence
- No feedback on whether it's helping or hurting

**Proposed Metric: Energy Margin**
```
energy_margin = band_energy - peak_floor

State:
  - margin > 0.05: peak_floor is OPTIMAL (hits ≈50% of the time, leaves headroom)
  - margin ≈ 0.01: peak_floor is TIGHT (almost every peak hits)
  - margin < 0.00: peak_floor is TOO HIGH (no beats detected)
  - margin > 0.15: peak_floor is TOO LOW (everything is a beat)

Action (per beat):
  - margin < 0.02 → RAISE peak_floor slightly (0.002 per beat)
  - margin 0.02-0.05 → NO CHANGE (optimal zone)
  - margin > 0.08 → LOWER peak_floor slightly (0.002 per beat)
  
Small step: 0.002 per beat = converges in ~50 beats (~30-60 seconds at 100 BPM)
```

**Why This Works:**
- Proportional feedback: further from optimal = bigger adjustment
- Continuous: responds to audio changes immediately
- Self-limiting: large margins naturally shrink, tight margins naturally widen
- Natural oscillation: settling point naturally centers at optimal margin

---

### 2. Sensitivity Optimization

**Current Behavior:**
- Cycles through pairs of threshold parameters (peak_decay, rise_sens)
- No clear feedback on what "right" sensitivity is
- Hunts for a target that isn't clearly defined

**Proposed Metric: Beat Consistency**
```
beat_consistency = |Average of last 5 beat intervals - Smoothed tempo|
                   ────────────────────────────────────────────────
                            Smoothed tempo

Interpretation:
  - 0.0-0.05 = EXCELLENT (very regular beats)
  - 0.05-0.15 = GOOD (acceptable rhythm)
  - 0.15-0.30 = POOR (scattered beats, false positives)
  - >0.30 = BAD (heartbeat mode, no useful rhythm)

Action (per beat after warmup):
  IF consistency > 0.15 (beats too scattered):
    → Raise threshold (peak_decay DOWN, rise_sens UP)
    → Step: 0.01 per beat until consistency < 0.12
  
  IF consistency < 0.05 (beats very tight):
    → Lower threshold (peak_decay UP, rise_sens DOWN)
    → Step: 0.005 per beat (slower, only fine-tune)
    
  Otherwise: hold steady
```

**Why This Works:**
- **Direct feedback:** You can see if audio is getting noisier or cleaner
- **Prevents false positives:** When scattered beat detection occurs, system tightens threshold
- **Doesn't over-adjust:** Once locked in, only fine-tunes
- **Tempo-agnostic:** Works at any BPM because it's ratio-based

---

### 3. Downbeat Confidence Optimization

**Current Behavior:**
- Hunt until 5 consecutive downbeats
- Downbeats determined by energy-based position tracking
- No optimization of how good the downbeat detection is

**Proposed Metric: Downbeat Energy Ratio**
```
downbeat_ratio = (strongest_position_energy / mean_of_all_positions)

Target zone: 1.8 - 2.2 (downbeat is clearly dominant)

Action (per downbeat):
  IF ratio > 2.2 (downbeat TOO pronounced, maybe real peak is earlier):
    → Lower peak_floor slightly (make detection more uniform)
    → Step: 0.001 per downbeat (slow tuning)
  
  IF ratio < 1.8 (downbeat not distinct enough):
    → Raise peak_floor slightly (enforce stricter energy threshold)
    → Step: 0.002 per downbeat
    
  Otherwise: LOCKED (ratio in sweet zone)

LOCKED status = no more adjustment for this parameter until ratio drifts out of zone
```

**Why This Works:**
- **Self-validating:** High ratio = strong downbeat pattern = confident detection
- **Passive:** Let audio decide when to unlock (if energy pattern changes)
- **Prevents squishing:** Stops adjusting once downbeat pattern is strong
- **No false positive creep:** If pattern degrades, system adjusts again

---

### 4. Audio Gain Optimization

**Current Behavior:**
- "Locks" after first beat
- Hunts by changing audio_amp (0.15-10.0 range)
- No clear target for what gain should be

**Proposed Metric: Intensity Mean**
```
intensity = band_energy / peak_envelope

Target: intensity_mean ≈ 0.3 - 0.4 (good separation for peaks, not saturated)

Track rolling average over last 16 beats:
  intensity_mean_recent = avg(intensity[...])

Action (per beat):
  IF intensity_mean_recent > 0.5 (audio too hot):
    → Lower audio_amp by step_size
    
  IF intensity_mean_recent < 0.2 (audio too quiet):
    → Raise audio_amp by step_size
    
  Otherwise: NO CHANGE
  
When LOCKED: still monitor intensity_mean
  IF intensity_mean drifts > 0.6 or < 0.1:
    → Resume HUNTING (audio changed significantly)
```

**Why This Works:**
- **Automatic level normalization:** System adjusts to keep signal in good zone
- **Adapts to audio loudness:** Different songs have different loudness; system normalizes
- **Re-hunts if needed:** If user plugs in quieter audio while locked, system adjusts
- **Simple target:** 0.3-0.4 is intuitive (30-40% of theoretical max)

---

## System Architecture (Pseudocode)

```python
class RealtimeAutoRanging:
    def __init__(self):
        self.target_energy_margin = 0.03      # 0.02-0.05 optimal zone
        self.target_downbeat_ratio = 2.0      # 1.8-2.2 optimal zone  
        self.target_intensity = 0.35         # 0.3-0.4 optimal zone
        self.target_beat_consistency = 0.10  # 0.05-0.15 optimal zone
        
        self.energy_margin_history = deque(maxlen=16)
        self.intensity_history = deque(maxlen=16)
        self.consistency_history = deque(maxlen=1)
        
        self.state = {
            'peak_floor': 'TUNING',     # Always active
            'sensitivity': 'TUNING',    # Always active (after tempo found)
            'audio_amp': 'TUNING',      # Always active
            'downbeat': 'MONITORING',   # Passive unless out of zone
        }
    
    def on_beat(self, event):
        """Called every beat - NO TIMER NEEDED"""
        
        # Metric 1: Peak Floor
        energy_margin = event.peak_energy - self.config.beat.peak_floor
        self.energy_margin_history.append(energy_margin)
        avg_margin = np.mean(self.energy_margin_history)
        
        if avg_margin < 0.02:
            self._adjust_param('peak_floor', +0.002)
        elif avg_margin > 0.08:
            self._adjust_param('peak_floor', -0.002)
        # else: stay in optimal zone
        
        # Metric 2: Audio Gain
        intensity = event.peak_energy / max(0.0001, self.peak_envelope)
        self.intensity_history.append(intensity)
        avg_intensity = np.mean(self.intensity_history)
        
        if avg_intensity > 0.5:
            self._adjust_param('audio_amp', -0.01)
        elif avg_intensity < 0.2:
            self._adjust_param('audio_amp', +0.01)
        
        # Metric 3: Beat Consistency (needs 5+ beats)
        if len(self.beat_intervals) >= 5:
            recent_cv = np.std(self.beat_intervals[-5:]) / np.mean(self.beat_intervals[-5:])
            
            if recent_cv > 0.15:  # Too scattered
                self._adjust_param('peak_decay', -0.01)      # Stricter
                self._adjust_param('rise_sens', +0.01)
            elif recent_cv < 0.05:  # Too tight (over-detecting)
                self._adjust_param('peak_decay', +0.005)     # Looser
                self._adjust_param('rise_sens', -0.005)
    
    def on_downbeat(self, event):
        """Called on downbeat - check ratio"""
        
        downbeat_ratio = event.downbeat_confidence  # Already computed by audio_engine
        
        if downbeat_ratio > 2.2:
            self._adjust_param('peak_floor', -0.001)
        elif downbeat_ratio < 1.8:
            self._adjust_param('peak_floor', +0.002)
        else:
            print(f"[AutoRange] Downbeat LOCKED at ratio={downbeat_ratio:.2f}")
```

---

## Migration Path

### Phase 1: Add Parallel System
- Keep existing timer-based hunting
- Launch new metric-based system in parallel
- Log both approaches' decisions
- Compare convergence speed and stability

### Phase 2: Switch Primary
- Make metric-based system primary for new users
- Keep timer system as fallback
- Add toggle: "Auto-Ranging Mode" (Classic vs. Real-Time)

### Phase 3: Full Replacement
- Remove timer-based system once metric system proven
- Simplify UI (no more hunting cycle display)
- Update AGENT_REFERENCE.md

---

## Expected Benefits

| Aspect | Current | Proposed |
|--------|---------|----------|
| **Convergence** | 100-200 beats | 20-50 beats |
| **Responsiveness** | Blind, batched | Real-time feedback |
| **CPU Cost** | Timer ticks even when nothing changes | Only on beats (lower overall) |
| **Manual Tuning** | Often needed after hunting | Rare (system self-corrects) |
| **Audio Changes** | Requires manual interrupt | Auto-adapts immediately |
| **UI Complexity** | HUNTING/REVERSING/LOCKED states | TUNING (ongoing) / LOCKED (stable) |

---

## Questions for Implementation

1. **When to declare "LOCKED"?**
   - Option A: When in optimal zone for 10+ consecutive beats
   - Option B: When adjustments become very small (<0.001)
   - Option C: Hybrid: stable for 5 beats AND in zone

2. **How to handle multiple param interactions?**
   - Example: Raising peak_floor affects both beat rate and downbeat ratio
   - Proposal: Single unified energy margin metric drives both

3. **Should warmup period exist?**
   - Keep brief 1-2 second warmup? (give system time to establish baseline)
   - Or start tuning immediately?

4. **Reverting parameter changes:**
   - If user manually adjusts a param, should system know and not revert it?
   - Simple approach: Lock that param when touched manually

5. **Emergency escape:**
   - Keep beathunting trigger for "zero beats after 3 cycles"?
   - Or let metric feedback naturally find solution?

---

## Next Steps

1. Review and approve proposal
2. Implement Metric 1 (Peak Floor) as proof of concept
3. A/B test vs. current system on various music
4. Iterate based on results
5. Implement remaining metrics
