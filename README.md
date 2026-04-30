# hebel

*Hebel* (הֶבֶל) is the Hebrew word for breath, vapor, the fleeting exhale.

Real-time wind synthesizer driven by an [OpenTheremin V4](https://www.gaudi.ch/OpenTheremin/). Wave your hands near the antennas: pitch hand sweeps the wind from low rumble to high whistle, volume hand goes from breeze to storm. No samples, no recordings. The sound is built from scratch and shaped by your gestures.

## Setup

You need the [OpenTheremin V4 MIDI firmware](https://github.com/MrDham/OpenTheremin_V4_with_MIDI) flashed on the Arduino. The DIN connector is optional; the script reads MIDI off the USB serial port.

Give yourself permission on the serial port:

```bash
sudo usermod -aG dialout $USER
# then log out and back in (or run: newgrp dialout)
```

Install the Python deps:

```bash
uv venv .venv
uv pip install --python .venv/bin/python mido python-rtmidi sounddevice numpy scipy pyserial
```

## Run

```bash
.venv/bin/python theremin_wind.py            # opens the TUI
.venv/bin/python theremin_wind.py --debug    # prints every MIDI message (no TUI)
.venv/bin/python theremin_wind.py --no-tui   # play without the TUI
.venv/bin/python theremin_wind.py --list     # show serial + audio devices
```

Press `q` to quit.

## The interface

```
─── theremin wind ──────────────────────────────
midi: /dev/ttyUSB0 @ 31250

features:  [x] 3-band (3)   [x] gust (g)

knobs:
 ▶ low band  Fc     110 Hz
   low band   Q      0.70
   ...
   drive             1.20

live:
  note= 60  bend=    +0  freq= 609.9 Hz
  amp=0.70  tilt=0.40  last_cc=7=89

↑↓ select knob   ←→ adjust   3/g toggle   q quit
```

Use `↑`/`↓` (or `j`/`k`) to pick a knob, `←`/`→` (or `h`/`l`) to nudge it. Press `3` or `g` to toggle the matching feature. Changes apply live with no restart.

## How the wind is built

Wind is several textures stacked together. The synth builds that stack and lets your theremin steer it.

1. **Pink noise** as the raw material. Pink noise has more energy in low frequencies than high ones, so it sounds darker and fuller than radio static. That alone gets us most of the way to "natural".

2. **A bandpass filter** that lets only a slice of the spectrum through. Your theremin pitch slides the slice up and down: low pitch gives a deep rumble, high pitch gives a thin whistle.

3. **Three bands at once**. Instead of one bandpass, the synth runs a fixed low rumble (about 110 Hz), a moving mid band that follows your theremin pitch, and a narrow high whistle (about 3.2 kHz). As your pitch goes up, the rumble fades and the whistle emerges. Toggle this off with `3` to compare.

4. **The volume hand** sets intensity. As loudness rises, the bandpass also narrows automatically, because real wind sharpens its character when it gets stronger, it doesn't just get louder.

5. **A gust LFO** gently swells the volume up and down on a 1 to 2 second cycle, so the wind feels alive even when your hands are still. Toggle with `g`.

6. **A Q drift** slowly wobbles the bandpass resonance on its own, since the theremin doesn't have a hand for Q. The texture's "personality" shifts subtly every few seconds, like changing weather behind your gestures. Set its depth to 0 if you want full hand control.

7. **A drive stage** at the end. At 1.2 it does nothing audible. Push it to 3 or 4 for warmth, or 8+ for the kind of distorted wind you'd hear in a horror film.

## Knobs

| Knob | Range | What it does |
|---|---|---|
| `low band Fc` | 20-500 Hz | Pitch of the constant rumble |
| `low band Q` | 0.3-5.0 | How focused the rumble is |
| `high band Fc` | 1-8 kHz | Pitch of the whistle that emerges at high theremin pitch |
| `high band Q` | 1-20 | Sharpness of the whistle (high = piercing, low = airy) |
| `mid Fc lo` | 100-800 Hz | Where the main wind starts at low theremin pitch |
| `mid Fc hi` | 0.8-5 kHz | Where the main wind arrives at high theremin pitch |
| `gust depth` | 0-1.0 | How strongly the wind breathes on its own |
| `gust tau` | 0.2-8 s | How slow that breathing is |
| `Q drift` | 0-0.5 | How much the texture drifts on its own |
| `Q drift tau` | 0.5-8 s | How slow the drift is |
| `drive` | 0.5-10 | Saturation: 1.2 transparent, 4 warm, 10 distorted |

## Under the hood

Audio runs in a `sounddevice` callback at 48 kHz, 512-sample blocks (~10.7 ms latency). The pink noise is Paul Kellet's six-pole IIR. Bandpass filters are RBJ "constant-skirt" biquads, recomputed per block so knob changes apply within ~10 ms. Both LFOs are one-pole low-passed Gaussian random walks normalized to unit variance, then `tanh`-clipped.

MIDI input is parsed inline from `/dev/ttyUSB0` at 31250 baud (running-status aware, sysex and real-time bytes ignored). No bridge software needed.

## Hardware notes

- **Port**: `/dev/ttyUSB0` (FT232 or CH340 USB-serial chip on the OpenTheremin).
- **Baud**: 31250. The MrDham firmware ships set to "true DIN MIDI" mode, so the bytes come out the USB cable at the standard MIDI wire speed even with no DIN connector wired. If you reflash with `Serial.begin(115200)`, run with `--baud 115200`.
- **Messages used**: channel 1, Note On/Off, Pitch Bend (default ±2 semitones), CC 1/7/11/74 for amplitude. Default firmware uses CC 7.
