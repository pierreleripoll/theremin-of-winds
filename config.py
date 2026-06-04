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

# Volume antenna polarity is "hand far = loud", so a hand pulled off the instrument
# sweeps the volume toward full right before the firmware goes quiet -- which used to
# fire a wind blast as people walked away. To stop that, amplitude RISES are slew-
# limited to this time constant (and only chase the target while fresh MIDI keeps
# arriving). Falls stay fast (SMOOTH_MS) so the wind can still be cut by hand. Pitch
# is unaffected. With this, a yanked hand never reaches full before input goes quiet.
AMP_RISE_S = 0.6        # seconds for the wind to swell toward a louder hand position

# Wind-machine restart inertia. After sitting idle the synth spins back up slowly,
# like a real fan overcoming inertia from a standstill -- the longer it was idle, the
# heavier the restart. Idle accumulates while nobody plays (capped at
# INERTIA_IDLE_FULL_S); on resume the presence spin-up time is ATTACK_S plus up to
# INERTIA_MAX_ADD_S of extra lag, scaled by how long it sat. Once it has spun up
# (presence reaches full) the stored inertia is spent and play is fully responsive
# again -- you can swing between amplitude extremes freely until the next long idle.
INERTIA_IDLE_FULL_S = 10.0   # idle this long -> maximum restart inertia
INERTIA_MAX_ADD_S = 1.2      # extra spin-up time constant added at full inertia (s)

# Rest-corner calibration. The instrument's resting posture (no hands near) sits at
# the corner of the control space: loudest volume + lowest pitch (both antennas read
# "hand far"), and the firmware keeps streaming it -- so without this the synth boots
# into a full-blast low wind. calibrate.py samples that corner (automatically
# ~CALIB_STARTUP_DELAY_S after launch -- step back while it does -- and on the TUI 'c'
# key); the audio gate then treats any reading within these margins of the corner as
# "nobody playing" and stays silent. To make sound the player must leave the corner:
# raise the pitch OR lower the volume. Only the extreme "lowest note at full blast"
# gesture is sacrificed (not a real playing gesture).
CALIB_STARTUP_DELAY_S = 2.0   # silence + wait this long after launch, then auto-sample
CALIB_SAMPLE_S = 1.5          # how long to watch the resting reading while sampling
REST_AMP_MARGIN = 0.07        # volume within this of the resting max still counts as rest
REST_NOTE_MARGIN = 3.0        # note within this many semitones of the resting low = rest

# Volume response curve. The theremin sends a 0..127 volume value; we raise the
# normalized value to this power before it becomes amplitude. >1 spends more of the
# input travel in the quiet region, so small hand movements give fine control over
# soft "breeze" levels and the loud range is reached only near the top. 1 = linear.
VOL_CURVE = 3.0

BAUD = 31250  # MrDham firmware in "true DIN MIDI" mode (Serial.begin(31250))

# --stream mode: the custom OpenTheremin V4 firmware (OpenThereminV4/, build.h
# SERIAL_PORT_MODE_STREAM) emits ASCII "P<pitch> V<vol> M<mute>\n" at ~200 Hz over
# 115200 baud instead of MIDI. pitch/vol are the firmware's linearized values, so
# we get continuous high-resolution control with no MIDI quantization. The full-scale
# constants map those raw ranges onto the synth's 0..1 control range and want tuning
# against the actual instrument (play the loudest / highest you reach -> set to that).
STREAM_BAUD = 115200
STREAM_PITCH_FULL = 8000.0    # streamPitch reading mapped to the top of the note range
STREAM_VOL_FULL = 4095.0      # streamVol reading mapped to full volume (firmware clamps here)

# Auto-play (demo) mode (--autoplay / 'a'): a background thread wanders the pitch
# and volume the way a hand would on the instrument, so the synth plays itself with
# no theremin. It eases between random "gesture" waypoints with occasional lulls, and
# drives them through State.fake_xy (which bumps msg_count, so the presence gate stays
# open). Meant for the exhibition: simulate someone playing without a person there.
AUTOPLAY_RATE_HZ = 30.0       # how often the simulated hand position updates
AUTOPLAY_MIN_GESTURE_S = 1.2  # shortest glide from one waypoint to the next
AUTOPLAY_MAX_GESTURE_S = 4.5  # longest glide (slow, sustained wind)
AUTOPLAY_PITCH_STEP = 0.45    # max change in pitch position (0..1) per gesture
AUTOPLAY_AMP_LO = 0.45        # quietest "playing" volume (pre vol-curve, 0..1)
AUTOPLAY_AMP_HI = 1.0         # loudest gust
AUTOPLAY_REST_PROB = 0.2      # chance a gesture is a near-silent lull instead
AUTOPLAY_REST_AMP = 0.08      # volume during a lull (eases the wind down, not fully out)

# DMX output (Enttec DMX USB Pro, opt-in via --dmx): drive curtain fans on a
# power dimmer while the synth makes sound. The dimmer (DDS-405) is addressed at
# DMX_CHANNEL, its four outlets landing on DMX_CHANNEL..DMX_CHANNEL+3.
DMX_BAUD = 57600        # FTDI side; the Pro generates DMX timing itself, so this is nominal
DMX_CHANNEL = 100       # dimmer base address: outlet 1 = this DMX channel (1..512)
DMX_ON_LEVEL = 255      # channel value at full (switch mode = on; dim mode = loudest wind)
DMX_FRAME_HZ = 40       # how often we refresh the DMX frame
DMX_RAMP_S = 0.25       # smooth the channel value toward its target (gentle on the dimmer)

# Two-tier fan engagement. Each stage integrates "loudness" into its own charge
# in [0,1] that rises (time constant attack_s) while sound_level >= loud_at and
# decays (decay_s) when it's quieter; the stage engages when its charge crosses
# engage_at and only releases once it falls back below disengage_at (hysteresis).
# This gives the "loud-ish for a moment, keeps blowing through a short stop"
# inertia, per outlet. Fields:
#   (label, outlet, loud_at, attack_s, decay_s, engage_at, disengage_at)
#   - "play"  : normal playing wakes the curtain          -> outlet 2 (DMX 101)
#   - "storm" : only sustained very loud wind adds a gust  -> outlet 4 (DMX 103)
# `outlet` is 1-based relative to DMX_CHANNEL. Outlets 1 and 3 are left dark.
DMX_STAGES = [
    ("play",  2, 0.08, 1.0, 3.0, 0.55, 0.20),
    ("storm", 4, 0.45, 4.0, 5.0, 0.85, 0.40),
]
# Proportional ("dim") mode: map wind intensity to fan speed. A fan on a triac
# dimmer won't start spinning below some voltage, so when engaged we never go
# below DMX_MIN_LEVEL; the loudest wind (sound level >= DMX_FULL_AT) reaches
# DMX_ON_LEVEL. Tune DMX_MIN_LEVEL with `dmx_test.py 100 <n>` to where your fan
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
