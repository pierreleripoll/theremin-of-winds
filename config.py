"""Synth tuning constants and defaults.

User-facing defaults live here. The TUI exposes most of these as live knobs;
CLI flags override the booleans. DSP-internal coefficients (pink-noise filter
poles etc.) live alongside their generator in `audio.py`.
"""

SR = 48000
BLOCK = 512                  # ~10.7 ms per callback
NOTE_LO, NOTE_HI = 36, 96
FREQ_LO, FREQ_HI = 150.0, 5000.0
RUMBLE_CUTOFF = 90.0
SMOOTH_MS = 12.0  # one-pole tau on target freq/amp; 60 ms felt laggy in --fake mode

# Auto-wind presence gate. The firmware STOPS sending MIDI when no hand is near the
# instrument, so "nobody playing" = no messages for IDLE_TIMEOUT_S. (While playing,
# the antennas always jitter, so messages keep flowing even for a held note.) When
# idle, a presence envelope releases over RELEASE_S so the sound fades out smoothly;
# the slow release also bridges brief gaps so a momentary dropout doesn't cut.
RELEASE_S = 1.5         # fade-out time once input stops (seconds; TUI knob)
IDLE_TIMEOUT_S = 0.4    # no MIDI for this long -> start fading
AMP_EPS = 1e-3          # snap presence to 0 below this (truly silent)

BAUD = 31250  # MrDham firmware in "true DIN MIDI" mode (Serial.begin(31250))

# 3-band synth: fixed low/high band centers, mid band tracks theremin pitch.
LOW_FC, LOW_Q = 110.0, 0.7
HIGH_FC, HIGH_Q = 3200.0, 6.0
MID_FC_LO, MID_FC_HI = 250.0, 2200.0   # mid band center range (Hz, log)

# Gust LFO: slow random-walk amplitude modulator.
GUST_DEPTH = 0.30          # ±30% amplitude swing
GUST_TAU_S = 1.6           # ~1.6 s correlation time

# Q drift: slow random-walk on bandpass resonance. Theremin doesn't control Q,
# so this fills a gap rather than competing with hand gestures.
Q_DRIFT_DEPTH = 0.15       # ±15% Q wobble
Q_DRIFT_TAU_S = 2.0        # ~2 s correlation time

# Drive: pre-tanh saturation gain. 1.2 = transparent (legacy), >3 = warm grit,
# >8 = aggressive overdrive ("horror" tone).
DRIVE = 1.2

# High-band gain (multiplier on the tilt^3 mix coef). 0 = no top-end "sizzle".
HIGH_BAND_GAIN = 0.45

# Mid-band Q amp ramp. mid_Q = 1.2 + MID_Q_MAX * amp^0.6, so larger -> louder
# play means narrower (more whistly) mid band. 0 disables the ramp entirely.
MID_Q_MAX = 4.0
