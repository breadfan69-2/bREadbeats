# bREadbeats v2.0

**Real-time music-reactive motion for Restim e-stim.**

bREadbeats listens to any audio playing on your PC and turns it into smooth, rhythmic motion — the way you'd naturally tap a finger or foot to music. Quiet passages produce subtle movement; the energy builds when the bass hits. Everything happens live, with no scripting required.

> **v2.0 is a major release.** The motion engine has been completely rebuilt from the ground up, delivering a level of musical responsiveness and timing accuracy not previously possible. See the [What's New in v2.0](#whats-new-in-v20) section for details.

---

## What Is Restim?

[Restim](https://github.com/diglet48/restim) is an open-source e-stim controller by diglet48 that accepts TCode commands over a TCP connection and drives compatible electro-stimulation hardware. bREadbeats acts as a live music-driven TCode source for Restim — you run both at the same time, and bREadbeats tells Restim what to do in sync with your audio.

---

## What bREadbeats Does

- **Captures live audio** from any Windows playback device using WASAPI loopback — your speakers, headphones, or any active output
- **Detects rhythm and tempo** in real time, locking onto BPM and tracking beats, downbeats, and musical energy
- **Generates orbital motion** that follows the music — timing, depth, and intensity all respond to what the audio is doing moment-to-moment
- **Sends TCode over TCP** to a running Restim instance

The motion orbits on a continuous arc, landing precisely on the beat — the way you'd tap a finger along to a song. Light taps on quiet sections, full extension when the bass is driving. The character of the motion changes with the music automatically.

---

## What's New in v2.0

Version 2.0 represents a complete rebuild of the intelligence that drives motion:

- **Orbital motion engine** — motion follows a smooth, continuous arc with beat-precise landing. No more linear back-and-forth.
- **Multi-band audio intelligence** — the system listens across bass, mid-bass, and upper frequency ranges simultaneously, making smarter decisions about when and how hard to move
- **Adaptive tempo lock** — a phase-locked BPM system that stabilizes quickly and stays locked through dynamic passages and quiet breaks
- **Expression layer** — long-session features including gradual intensity ramping, orbit variation, subtle center drift, and tension pauses that respond to energy drops in the music
- **Baked motion profile** — ships with a tuned motion intelligence baseline that works well across a wide range of music without any configuration

Everything is automatic. Put on music, press Start and Play, and it works.

---

## Requirements

- Windows 10 or 11 (64-bit)
- [Restim](https://github.com/diglet48/restim) running and connected to your device
- A compatible e-stim device supported by Restim
- Audio playing on your PC (music, video, game audio — anything)

---

## Quick Start (EXE)

1. **Launch Restim** first and connect your device
2. **Launch** `bREadbeats.exe`
3. **Set your audio device** — `Options → Audio Device` — select the output you're listening through (e.g. your speakers or headphones listed as a loopback source)
4. **Set the connection** — `Options → Connection Settings` — defaults (`127.0.0.1 : 12347`) work if Restim is on the same machine
5. **Click `▶ Start`** to begin audio capture
6. **Click `▶ Play`** to enable motion output
7. **Play some music** and watch the beat indicators light up

> Both **Start** and **Play** must be active for motion to be sent.

---

## Files Bundled with the EXE

The EXE is self-contained — no Python or additional installs needed. On first run it creates a few files next to the EXE:

| File/Folder | Purpose |
|---|---|
| `config.json` | All your settings — saved automatically, safe to back up |
| `breadbeats.log` | Startup and runtime log — first place to look if something goes wrong |
| `defaults/learning/` | Bundled motion intelligence baseline (written on first run) |
| `presets.json` | Saved presets (created when you save a preset via the Presets menu) |

---

## Connecting to Restim

| Setting | Default |
|---|---|
| Host | `127.0.0.1` |
| Port | `12347` |

To change: `Options → Connection Settings`

Restim must be running and have a device connected before you click Play. bREadbeats will show **"Connected"** in green once the TCP link is up. If it shows **"Disconnected"**, verify Restim is open and TCode input is enabled in its settings.

---

## What You Should See When It's Working

| Indicator | What it Means |
|---|---|
| Waveform moving in the visualizer | Audio is being captured |
| **●** Beat dot (left) flashing in time | Beats are being detected |
| **●** Downbeat dot (middle) flashing on beat 1 | Downbeat pattern is active |
| **●** Metronome dot (right): yellow → green | BPM locking in; green = locked, motion is live |
| BPM counter stable | Tempo confirmed |
| Orbital dot moving on the position display | Motion is being generated and sent to Restim |
| **"Connected"** status in green | Restim TCP link is live |

**Healthy state:** all three indicator dots blink rhythmically, the metronome dot is green, and the position display shows the dot tracing the arc in time with the music.

---

## If Nothing Is Moving

Work through these in order:

### 1. Check your audio device
`Options → Audio Device` — the device must match what you're actually hearing audio on.
The waveform at the top of the window should be moving if audio is being captured.

### 2. Confirm Start and Play are both active
Both buttons must be pressed. They toggle — if you clicked twice accidentally, press once more.

### 3. Wait for the metronome to lock
The right indicator turns **yellow** while locking, **green** when confirmed. Most tracks lock within 4–8 beats. Motion won't start until the lock is green.

### 4. Raise the sensitivity
In **Beat Detection**:
- Increase **Amplification**
- Increase **Flux Multiplier**
- Lower **Depth** all the way to 0 to reset it

### 5. Lower the volume/motion threshold
The **volume/motion threshold** slider in Main Controls sets the minimum RMS level required for motion. Lower it if the signal is quiet.

### 6. Check the Restim connection
Confirm **"Connected"** appears in green. If not, check Restim is open and accepting TCode input on port 12347.

---

## If There Is Too Much Motion or Jitter

- Lower **Amplification** and **Flux Multiplier** in Beat Detection
- Reduce **Sensitivity** slightly — small changes have a big effect
- `Options → Effects → Disable Jitter` to turn off the texture layer
- Raise the **volume/motion threshold** to filter noise between beats

---

## Controls Overview

### Main Controls (always visible)

| Control | What it Does |
|---|---|
| **Sensitivity** | Core motion gate — the most impactful single control. Higher = more responsive to quieter beats. |
| **volume/motion threshold** | Silence gate — below this RMS level, motion stops entirely |
| **Motion Ramp** | Gradually increases intensity over a session (0 = off). Good for longer sessions. |
| **Pulse Settings** | Opens the pulse frequency settings popout |
| **Volume** | Output volume level sent to Restim |

### Beat Detection tab

| Control | What it Does |
|---|---|
| **Amplification** | Pre-gain applied to the audio signal before analysis |
| **Sensitivity** | Beat trigger threshold |
| **Depth** | High values suppress quiet beats — lower toward 0 to allow more |
| **Peak decay** | How quickly the beat envelope falls between hits |
| **Rise sensitivity** | Sensitivity to onset transients (attack detection) |
| **Flux Multiplier** | Scales the spectral flux signal — raise if beats are being missed |
| **Detection freq range** | Frequency band used for beat detection — bass-focused is typical |

### Tempo Tracking tab

Controls for BPM stability, phase snap aggressiveness, and lock conditions. Leave at defaults unless the BPM display is erratic.

### Advanced Controls

Multi-band signal gate tuning. Not needed for typical use — these are for dialing in specific response behavior on unusual audio sources or genres.

---

## Genre and Audio Type Notes

bREadbeats works best with music that has a clear, consistent rhythm:

| Audio Type | Expected Behavior |
|---|---|
| Electronic / EDM / Dance | Excellent — strong kick and consistent BPM lock quickly |
| Pop / Rock with drums | Very good — kick and snare give reliable beat detection |
| Hip-hop / R&B | Good — may need Flux Multiplier raised for low-tempo tracks |
| Classical / Orchestral | Variable — complex dynamics; try lowering peak floor and increasing rise sensitivity |
| Ambient / Drone | Limited — no clear beat; motion will be subtle and irregular |
| Game audio / Film scores | Depends on the track — works well when action cues have rhythmic content |
| Podcasts / Speech | Not designed for this use case |

Tracks with a prominent bass or kick drum consistently give the best results.

---

## Presets

`Options → Presets` — save and load all current settings as a named preset. Useful for bookmarking a tuned configuration per genre, device, or mood.

---

## Logs and Troubleshooting

Runtime log: `breadbeats.log` (next to the EXE)

Check this file first if the app doesn't behave as expected. It captures startup events, connection attempts, and runtime warnings.

For more output during troubleshooting, set this environment variable before launching:
```
BREADBEATS_DEBUG_STDIO=1
```

---

## Known Limitations

- **Windows only** — WASAPI loopback is a Windows-specific audio API; no Mac or Linux support
- **Requires Restim** — bREadbeats is a controller, not a standalone device driver
- **No Bluetooth audio loopback** — if your audio output is Bluetooth, loopback capture may not be available; use a wired output or a virtual audio cable
- **High-BPM tracks (>160 BPM)** — the system may halve the detected BPM on very fast material; this is intentional to maintain comfortable motion timing
- **Very quiet or highly compressed audio** — may require raising Amplification significantly or lowering the silence gate threshold

---

## Frequently Asked Questions

**Why does motion stop during quiet sections?**  
This is intentional. The silence gate pauses motion when the audio drops below the volume/motion threshold, then resumes when the signal returns. Lower the threshold if you want motion during softer passages.

**The BPM looks wrong / keeps jumping around.**  
Try raising the Stability Threshold in the Tempo Tracking tab, or lower the audio Amplification if the signal is very loud and clipping. Some tracks with irregular timing will stabilize after a few bars.

**There's motion but it feels out of sync.**  
Wait for the metronome indicator to turn green — motion is intentionally gated until BPM is confirmed. If it stays yellow, raise the Flux Multiplier or lower Stability Threshold.

**Why does it pick up half or double the actual BPM?**  
At extreme tempos the system may halve or double to keep motion in a comfortable range. This is expected behavior, not an error.

**Can I use a microphone instead of loopback?**  
Yes — select a microphone input from `Options → Audio Device`. Results vary depending on room acoustics and mic placement.

**Do my settings save automatically?**  
Yes — settings are written to `config.json` when you close the app. You can also save named presets via `Options → Presets`.

**Where do I find the log file?**  
`breadbeats.log` in the same folder as `bREadbeats.exe`.

---

## Tips

- WASAPI loopback is the ideal audio source — it captures exactly what you hear with no extra hardware
- Music with a clearly audible kick drum locks fastest and tracks most accurately
- Motion naturally varies with the song — quiet sections producing less is by design, not a problem
- The Motion Ramp feature in Main Controls lets intensity build gradually over a session rather than starting at full power
- Settings are auto-saved on close; use Presets to snapshot configurations you want to return to

---

## What's New in v2.0

| Area | v1.x | v2.0 |
|---|---|---|
| Motion type | Linear stroke | Continuous orbital arc |
| Beat timing | Approximate | Phase-locked, beat-precise landing |
| Audio analysis | Single band | Multi-band simultaneous analysis |
| Tempo tracking | Basic BPM detection | PLL-based metronome with stability gating |
| Motion intelligence | Rule-based | Baked adaptive profile (ships with EXE) |
| Expression | None | Orbit variation, wander, tension pauses, session arc |
| Silence handling | Hard cut | Graceful fade with post-silence ramp |

---

## Credits & Acknowledgements

- **digitalparkingleot** — original concept inspirations
- **edger477** — funscript tooling ideas
- **diglet48** — [Restim](https://github.com/diglet48/restim) — the platform that makes this possible
- **shadlock0133** — music-vibes

---

## Support the Project

If you enjoy bREadbeats:  
👉 [https://ko-fi.com/breadbeats](https://ko-fi.com/breadbeats)

Bug reports & preset sharing: **bREadfan_69@hotmail.com**

---

## License

See [LICENSE](LICENSE)
