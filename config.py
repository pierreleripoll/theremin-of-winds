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
# the slow release also bridges brief gaps so a momentary dropout doesn't cut. When
# input resumes the envelope attacks over ATTACK_S, so the sound spins up from zero
# like a wind turbine rather than popping on.
ATTACK_S = 0.4          # spin-up time when input resumes (seconds; TUI knob)
RELEASE_S = 0.6         # fade-out time once input stops (seconds; TUI knob)
IDLE_TIMEOUT_S = 0.4    # no MIDI for this long -> start fading
AMP_EPS = 1e-3          # snap presence to 0 below this (truly silent)

# Volume response curve. The theremin sends a 0..127 volume value; we raise the
# normalized value to this power before it becomes amplitude. >1 spends more of the
# input travel in the quiet region, so small hand movements give fine control over
# soft "breeze" levels and the loud range is reached only near the top. 1 = linear.
VOL_CURVE = 3.0

BAUD = 31250  # MrDham firmware in "true DIN MIDI" mode (Serial.begin(31250))

# DMX output (Enttec DMX USB Pro, opt-in via --dmx): drive a fan on a power dimmer
# while the synth makes sound. The dimmer is addressed at channel 50.
DMX_BAUD = 57600        # FTDI side; the Pro generates DMX timing itself, so this is nominal
DMX_CHANNEL = 50        # dimmer/fan address (1..512)
DMX_ON_LEVEL = 255      # channel value at full (switch mode = on; dim mode = loudest wind)
DMX_FRAME_HZ = 40       # how often we refresh the DMX frame
DMX_SOUND_EPS = 5e-3    # gated output amplitude above this counts as "making sound"
DMX_HOLD_S = 0.5        # keep the fan on this long after sound stops (anti-flicker)
DMX_RAMP_S = 0.25       # smooth the channel value toward its target (gentle on the dimmer)
# Proportional ("dim") mode: map wind intensity to fan speed. A fan on a triac
# dimmer won't start spinning below some voltage, so when there's sound we never
# go below DMX_MIN_LEVEL; the loudest wind (sound level >= DMX_FULL_AT) reaches
# DMX_ON_LEVEL. Tune DMX_MIN_LEVEL with `dmx_test.py 50 <n>` to where your fan
# reliably starts.
DMX_MIN_LEVEL = 80      # lowest level that still spins the fan (0..255)
DMX_FULL_AT = 0.7       # sound level that maps to DMX_ON_LEVEL

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

# Pipe organ mode (toggle with 'o'): an additive harmonic stack (the way an organ
# stacks pipe ranks) blended with pink noise resonating in the "tube", so it sounds
# like wind howling through metal pipes (aeolian) rather than a clean electronic organ.
ORGAN_OCTAVE = -2.0        # octave of the lowest organ note (-3 = very deep pedal bass)
ORGAN_BRIGHTNESS = 0.35    # tilts the upper partials: 0 = dark/bassy, 1 = full bright chorus
ORGAN_AIR = 0.45           # resonant "air in the metal tube" (pitched breath through the pipe)
ORGAN_WIND = 0.7           # how strongly the wind's gusting drives the organ (swell/bright/pitch)
# The organ is a background colour, not a lead voice: it should stay discreet and
# only rise when the wind is strong AND has been strong for a while (inertia). A slow
# envelope ("swell") follows how far the wind exceeds ORGAN_THRESHOLD, building over
# ORGAN_RISE_S and fading over ORGAN_FALL_S, and gates the whole organ. Below the
# threshold the organ never wakes; ORGAN_LEVEL scales its overall presence.
ORGAN_LEVEL = 0.55         # overall organ loudness relative to the wind (lower = more discreet)
ORGAN_THRESHOLD = 0.55     # wind intensity (0..1) the swell must exceed before the organ builds
ORGAN_RISE_S = 6.0         # how long strong wind must hold before the organ is fully present
ORGAN_FALL_S = 2.5         # how long the organ lingers after the wind eases
TREM_DEPTH = 0.05          # gentle periodic tremulant on top of the swell (amplitude)
TREM_RATE = 5.5            # tremulant rate (Hz; classic organ tremulant is 5-6 Hz)
TREM_PITCH = 0.003         # periodic pitch vibrato, fractional (~5 cents)
