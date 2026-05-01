#!/usr/bin/env python3
"""
Real-time wind-noise synth driven by an OpenTheremin V4 (MIDI firmware).

The OpenTheremin V4 with the MrDham/Vincent Dhamelincourt MIDI firmware sends
raw MIDI bytes over USB serial @ 115200 baud (it is NOT a USB-MIDI device).
This script reads that serial stream directly, parses MIDI inline, and drives
a small real-time wind synth (band-passed white + low-passed brown noise).

Default mapping (matches the firmware's defaults: ch 1, pitch-bend on, loop
antenna -> CC7):
  - Note On + Pitch Bend  -> bandpass center frequency  (wind "speed")
  - CC 1 / 7 / 11 / 74    -> output amplitude           (wind "intensity")
  - Note On velocity      -> initial amplitude until a CC arrives
  - Note Off              -> amplitude target -> 0 (smooth release)

Usage:
  python theremin_wind.py --list                  # list serial + audio devices
  python theremin_wind.py                         # auto-pick /dev/ttyUSB* or ACM*
  python theremin_wind.py --serial /dev/ttyUSB0
  python theremin_wind.py --debug                 # print every MIDI msg
"""
import argparse
import glob
import math
import sys
import threading
import time

import numpy as np
import serial
import sounddevice as sd
from scipy.signal import iirfilter, lfilter, lfilter_zi

SR = 48000
BLOCK = 512                  # ~10.7 ms per callback
NOTE_LO, NOTE_HI = 36, 96
FREQ_LO, FREQ_HI = 150.0, 5000.0
RUMBLE_CUTOFF = 90.0
SMOOTH_MS = 12.0  # one-pole tau on target freq/amp; 60 ms felt laggy in --fake mode
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

# Paul Kellet's refined pink noise filter (6 parallel one-poles + white passthrough).
# Sounds smoother than Voss-McCartney; cheap when run via lfilter per pole.
PINK_POLES = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616]
PINK_GAINS = [0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980]
PINK_DIRECT = 0.5362
PINK_SCALE = 0.11


# ---- shared state ---------------------------------------------------------

class State:
    def __init__(self):
        self.lock = threading.Lock()
        # MIDI-driven
        self.target_freq = 800.0
        self.target_amp = 0.0
        self.cur_freq = 800.0
        self.cur_amp = 0.0
        self.note: int | None = None
        self.pitch_bend = 0  # signed: -8192..+8191
        self.last_cc: tuple[int, int] | None = None
        # feature toggles (TUI-editable)
        self.use_3band = True
        self.use_gust = True
        self.use_fifth = False
        self.third_mode = 0  # 0=off, 1=minor (6:5), 2=major (5:4)
        # knobs (TUI-editable)
        self.low_fc = LOW_FC
        self.low_q = LOW_Q
        self.high_fc = HIGH_FC
        self.high_q = HIGH_Q
        self.mid_fc_lo = MID_FC_LO
        self.mid_fc_hi = MID_FC_HI
        self.gust_depth = GUST_DEPTH
        self.gust_tau_s = GUST_TAU_S
        self.q_drift_depth = Q_DRIFT_DEPTH
        self.q_drift_tau_s = Q_DRIFT_TAU_S
        self.drive = DRIVE
        self.high_band_gain = HIGH_BAND_GAIN
        self.mid_q_max = MID_Q_MAX
        self.tone_level = 0.0  # bourdon voice (pitched-wind intervals) added on top of wind
        self.bourdon_q = 12.0  # higher = narrower whistle, more pitched; lower = airier
        # spatial mode: pitch antenna keeps its normal pitch role; volume antenna (CC)
        # drives stereo pan position. Amplitude stays at the note-on velocity level.
        self.spatial_mode = False
        self.target_position = 0.5  # 0..1, 0 = full left, 1 = full right
        self.cur_position = 0.5
        self.pan_floor = 0.15  # minimum gain on the "off" side; 0 = hard pan, 0.5 = barely panned
        # macros (preset sliders): when adjusted, write through to fine knobs above.
        # Defaults of 1.0 reproduce the historical "stormy" sound.
        self.brightness = 1.0
        self.whistle = 1.0

    def apply_brightness(self):
        # 0 = dark/calm, 1 = bright/stormy. Maps to high_band_gain and drive.
        b = self.brightness
        self.high_band_gain = HIGH_BAND_GAIN * b
        self.drive = 0.5 + (DRIVE - 0.5) * b

    def apply_whistle(self):
        # 0 = no resonance/howl, 1 = full whistle ramp. Maps to high_q and mid_q_max.
        w = self.whistle
        self.high_q = 1.0 + (HIGH_Q - 1.0) * w
        self.mid_q_max = MID_Q_MAX * w

    def _recompute_freq(self):
        if self.note is None:
            return
        bend_semis = self.pitch_bend / 8192.0 * 2.0  # default ±2 semitones
        n = max(NOTE_LO, min(NOTE_HI, self.note + bend_semis))
        ratio = (n - NOTE_LO) / (NOTE_HI - NOTE_LO)
        self.target_freq = FREQ_LO * (FREQ_HI / FREQ_LO) ** ratio

    def note_on(self, note: int, vel: int):
        self.note = note
        self._recompute_freq()
        if self.target_amp == 0.0:
            self.target_amp = (vel / 127.0) ** 1.8

    def note_off(self, note: int):
        if self.note == note:
            self.note = None
            self.target_amp = 0.0

    def pitch_wheel(self, lsb: int, msb: int):
        self.pitch_bend = ((msb << 7) | lsb) - 8192
        self._recompute_freq()

    def cc(self, num: int, val: int):
        self.last_cc = (num, val)
        if num in (1, 7, 11, 74):
            if self.spatial_mode:
                # left hand = pan; amp stays at its note-on value.
                self.target_position = val / 127.0
            else:
                self.target_amp = (val / 127.0) ** 1.8

    def fake_xy(self, x_norm: float, y_norm: float):
        """Trackpad fake-input: x ∈ [0,1] → freq (log). Y is amp normally;
        in spatial mode Y → pan position and amp is held constant while touching."""
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        self.target_freq = FREQ_LO * (FREQ_HI / FREQ_LO) ** x_norm
        if self.spatial_mode:
            self.target_position = y_norm
            self.target_amp = 0.7
        else:
            self.target_amp = y_norm ** 1.8


# ---- MIDI-over-serial parser ---------------------------------------------

class MidiSerialReader:
    """Streaming MIDI byte parser. Handles running status; ignores sysex
    and 0xF8+ real-time bytes."""

    # status nibble -> number of data bytes
    DATA_LEN = {
        0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2,
        0xC0: 1, 0xD0: 1, 0xE0: 2,
    }

    def __init__(self, state: State, debug: bool):
        self.state = state
        self.debug = debug
        self.status = 0
        self.data: list[int] = []
        self.in_sysex = False

    def _dispatch(self, status: int, data: list[int]):
        kind = status & 0xF0
        ch = status & 0x0F
        if self.debug:
            print(f"[midi] ch{ch+1:>2} {kind:#04x} {data}")
        with self.state.lock:
            if kind == 0x90 and data[1] > 0:
                self.state.note_on(data[0], data[1])
            elif kind == 0x80 or (kind == 0x90 and data[1] == 0):
                self.state.note_off(data[0])
            elif kind == 0xE0:
                self.state.pitch_wheel(data[0], data[1])
            elif kind == 0xB0:
                self.state.cc(data[0], data[1])
            # 0xA0 (poly AT), 0xC0 (PC), 0xD0 (chan AT) ignored

    def feed(self, byte: int):
        # System real-time: interleavable, ignore
        if 0xF8 <= byte <= 0xFF:
            return

        # Sysex framing
        if byte == 0xF0:
            self.in_sysex = True
            return
        if byte == 0xF7:
            self.in_sysex = False
            return
        if self.in_sysex:
            return

        # System common (0xF1..0xF6) — clears running status, ignore
        if 0xF1 <= byte <= 0xF6:
            self.status = 0
            self.data = []
            return

        if byte & 0x80:
            # Channel status byte
            self.status = byte
            self.data = []
            return

        # Data byte. Need a valid running status.
        if not self.status:
            return
        self.data.append(byte)
        need = self.DATA_LEN.get(self.status & 0xF0, 0)
        if need and len(self.data) >= need:
            d = self.data[:need]
            self.data = []
            self._dispatch(self.status, d)


def find_touchpad():
    """Return the first evdev device that looks like a touchpad, or None.

    evdev.list_devices() silently drops devices the current user can't read,
    so an empty result usually means a missing 'input' group, not "no devices".
    Detect that and raise PermissionError so the caller can show a useful hint.
    """
    import evdev
    from evdev import ecodes
    paths = evdev.list_devices()
    if not paths and glob.glob("/dev/input/event*"):
        raise PermissionError("/dev/input/event* exists but is unreadable by this user")
    for path in paths:
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        caps = dev.capabilities()
        abs_codes = {c[0] for c in caps.get(ecodes.EV_ABS, [])}
        key_codes = set(caps.get(ecodes.EV_KEY, []))
        if (ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
                and ecodes.BTN_TOUCH in key_codes
                and ecodes.INPUT_PROP_POINTER in dev.input_props()):
            return dev
        dev.close()
    return None


def trackpad_loop(state: State, dev, grab: bool = False):
    """Read absolute X/Y from a touchpad and drive State.fake_xy.

    Trackpad space → synth: x_min..x_max → freq (log), y_min..y_max → amp
    (inverted so top-of-pad = loud). Lifting the finger drops amp to 0.
    """
    from evdev import ecodes
    ax = dev.absinfo(ecodes.ABS_X)
    ay = dev.absinfo(ecodes.ABS_Y)
    x_min, x_max = ax.min, ax.max
    y_min, y_max = ay.min, ay.max
    x_span = max(1, x_max - x_min)
    y_span = max(1, y_max - y_min)

    cur_x = (x_min + x_max) / 2
    cur_y = (y_min + y_max) / 2
    touching = False

    if grab:
        dev.grab()
    try:
        for ev in dev.read_loop():
            if ev.type == ecodes.EV_ABS:
                if ev.code == ecodes.ABS_X:
                    cur_x = ev.value
                elif ev.code == ecodes.ABS_Y:
                    cur_y = ev.value
            elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_TOUCH:
                touching = bool(ev.value)
                if not touching:
                    with state.lock:
                        state.target_amp = 0.0
            elif ev.type == ecodes.EV_SYN and touching:
                x_norm = (cur_x - x_min) / x_span
                y_norm = 1.0 - (cur_y - y_min) / y_span
                with state.lock:
                    state.fake_xy(x_norm, y_norm)
    finally:
        if grab:
            try:
                dev.ungrab()
            except OSError:
                pass


def serial_loop(state: State, port: str, baud: int, debug: bool):
    print(f"[midi] opening serial: {port} @ {baud}")
    while True:
        try:
            with serial.Serial(port, baud, timeout=0.05) as s:
                print("[midi] reading…")
                reader = MidiSerialReader(state, debug)
                while True:
                    chunk = s.read(64)
                    for b in chunk:
                        reader.feed(b)
        except serial.SerialException as e:
            print(f"[midi] serial error: {e}; retry in 2s", file=sys.stderr)
            time.sleep(2.0)


# ---- audio synth ---------------------------------------------------------

def pan_gains(position: float, n_channels: int, floor: float = 0.0) -> tuple[float, ...]:
    """Per-channel gains for a mono source panned to `position` ∈ [0, 1].

    Stereo: equal-power L/R pan (cos/sin), then clamped from below by `floor`
    so neither earbud ever goes silent at extreme pan. Surround layouts
    (quad / 5.1) would extend this with VBAP over a speaker-angle table.
    """
    p = max(0.0, min(1.0, position))
    if n_channels == 2:
        return (max(floor, math.cos(p * math.pi / 2.0)),
                max(floor, math.sin(p * math.pi / 2.0)))
    return tuple(1.0 / math.sqrt(n_channels) for _ in range(n_channels))


def build_biquad_bandpass(fc: float, Q: float, sr: int):
    """RBJ bandpass (constant skirt, peak gain = Q)."""
    w0 = 2.0 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * Q)
    b = np.array([alpha, 0.0, -alpha], dtype=np.float64)
    a = np.array([1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha], dtype=np.float64)
    return b / a[0], a / a[0]


def make_audio_callback(state: State):
    """Audio callback. Reads feature flags + knobs from State each block, so a
    TUI thread can mutate them live."""
    # --- filters whose coefs are static (legacy single-band rumble) ---
    rumble_b, rumble_a = iirfilter(
        2, RUMBLE_CUTOFF / (SR / 2.0), btype="low", ftype="butter"
    )
    rumble_zi = lfilter_zi(rumble_b, rumble_a) * 0.0
    bp_zi = np.zeros(2)

    # --- 3-band biquad states ---
    low_zi = np.zeros(2)
    mid_zi = np.zeros(2)
    high_zi = np.zeros(2)

    # --- pink noise state (one zi per pole) ---
    pink_zis = [np.zeros(1) for _ in PINK_POLES]

    # --- gust + Q-drift LFO state (block-rate one-pole random walks) ---
    gust_state = 0.0
    q_drift_state = 0.0

    # --- bourdon voices: narrow bandpass on the same pink noise. zi persists across blocks
    # so coef changes (player moves the theremin) don't click. ---
    bourd_root_zi = np.zeros(2)
    bourd_fifth_zi = np.zeros(2)
    bourd_third_zi = np.zeros(2)

    # control smoothing
    tau_blocks = (SMOOTH_MS / 1000.0) * SR / BLOCK
    alpha_smooth = 1.0 - math.exp(-1.0 / max(tau_blocks, 1.0))

    rng = np.random.default_rng()

    def callback(outdata, frames, time_info, status):
        nonlocal rumble_zi, bp_zi, low_zi, mid_zi, high_zi, gust_state, q_drift_state
        nonlocal bourd_root_zi, bourd_fifth_zi, bourd_third_zi
        if status:
            print(f"[audio] {status}", file=sys.stderr)

        with state.lock:
            tf, ta = state.target_freq, state.target_amp
            use_3band = state.use_3band
            use_gust = state.use_gust
            use_fifth = state.use_fifth
            third_mode = state.third_mode
            tone_level = state.tone_level
            bourdon_q = state.bourdon_q
            spatial_mode = state.spatial_mode
            tp = state.target_position
            pan_floor = state.pan_floor
            low_fc, low_q = state.low_fc, state.low_q
            high_fc, high_q = state.high_fc, state.high_q
            mid_fc_lo, mid_fc_hi = state.mid_fc_lo, state.mid_fc_hi
            gust_depth, gust_tau_s = state.gust_depth, state.gust_tau_s
            q_drift_depth, q_drift_tau_s = state.q_drift_depth, state.q_drift_tau_s
            drive = state.drive
            high_band_gain = state.high_band_gain
            mid_q_max = state.mid_q_max

        state.cur_freq += (tf - state.cur_freq) * alpha_smooth
        state.cur_amp += (ta - state.cur_amp) * alpha_smooth
        state.cur_position += (tp - state.cur_position) * alpha_smooth
        f = max(60.0, min(SR * 0.45, state.cur_freq))
        amp = max(0.0, min(1.0, state.cur_amp))

        # --- pink noise source (Paul Kellet's 6-pole IIR + white passthrough) ---
        white = rng.standard_normal(frames).astype(np.float64) * 0.4
        src = white * PINK_DIRECT
        for i, (pole, gain) in enumerate(zip(PINK_POLES, PINK_GAINS)):
            y, pink_zis[i] = lfilter([gain], [1.0, -pole], white, zi=pink_zis[i])
            src = src + y
        src *= PINK_SCALE

        # --- gust LFO: unit-variance random walk -> tanh -> ±depth around 1.0 ---
        if use_gust:
            gust_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(gust_tau_s, 0.05))
            gust_state = (1.0 - gust_alpha) * gust_state + gust_alpha * rng.standard_normal()
            # one-pole on unit-variance noise has stationary stddev sqrt(a/(2-a));
            # rescale so gust_norm has unit variance and `depth` is meaningful.
            norm = math.sqrt((2.0 - gust_alpha) / gust_alpha)
            gust_mod = 1.0 + gust_depth * math.tanh(gust_state * norm * 0.7)
        else:
            gust_mod = 1.0
        amp_eff = amp * gust_mod

        # --- Q drift: same shape as gust, but on resonance (no theremin control over Q) ---
        if q_drift_depth > 0.0:
            q_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(q_drift_tau_s, 0.05))
            q_drift_state = (1.0 - q_alpha) * q_drift_state + q_alpha * rng.standard_normal()
            q_norm = math.sqrt((2.0 - q_alpha) / q_alpha)
            q_mod = 1.0 + q_drift_depth * math.tanh(q_drift_state * q_norm * 0.7)
        else:
            q_mod = 1.0

        # --- synthesis ---
        if use_3band:
            tilt = (math.log(f) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))
            tilt = max(0.0, min(1.0, tilt))

            mid_fc = mid_fc_lo * (mid_fc_hi / mid_fc_lo) ** tilt
            mid_Q = (1.2 + mid_q_max * (amp ** 0.6)) * q_mod

            # rebuild biquads each block — cheap and lets knobs change live
            low_b, low_a = build_biquad_bandpass(low_fc, low_q, SR)
            mid_b, mid_a = build_biquad_bandpass(mid_fc, mid_Q, SR)
            high_b, high_a = build_biquad_bandpass(high_fc, high_q, SR)

            low, low_zi = lfilter(low_b, low_a, src, zi=low_zi)
            mid, mid_zi = lfilter(mid_b, mid_a, src, zi=mid_zi)
            high, high_zi = lfilter(high_b, high_a, src, zi=high_zi)

            g_low = (1.0 - tilt) * 0.6 + 0.3
            g_mid = 0.7
            g_high = (tilt ** 3) * high_band_gain

            mix = (low * g_low * 4.0 + mid * g_mid * 1.5 + high * g_high * 1.5) * amp_eff
        else:
            Q = (1.2 + (mid_q_max + 1.0) * (amp ** 0.6)) * q_mod
            b, a = build_biquad_bandpass(f, Q, SR)
            bp, bp_zi = lfilter(b, a, src, zi=bp_zi)
            rumble, rumble_zi = lfilter(rumble_b, rumble_a, src, zi=rumble_zi)
            mix = (bp * 1.5 * 0.75 + rumble * 4.0 * 0.55) * amp_eff

        # --- bourdon: narrow bandpass on the same pink noise at root + optional 5th/3rd.
        # Pitched-wind whistle (like air across a bottle), not a sine pad. Tracks the theremin
        # in parallel — the singer-and-shadowing-monk effect of medieval organum. ---
        if tone_level > 0.0:
            br_b, br_a = build_biquad_bandpass(f, bourdon_q, SR)
            voices, bourd_root_zi = lfilter(br_b, br_a, src, zi=bourd_root_zi)
            n_voices = 1

            if use_fifth:
                f5 = f * 1.5
                if f5 < SR * 0.45:
                    bf_b, bf_a = build_biquad_bandpass(f5, bourdon_q, SR)
                    fifth, bourd_fifth_zi = lfilter(bf_b, bf_a, src, zi=bourd_fifth_zi)
                    voices = voices + fifth
                    n_voices += 1

            if third_mode > 0:
                ratio = 1.2 if third_mode == 1 else 1.25  # 6:5 minor, 5:4 major
                f3 = f * ratio
                if f3 < SR * 0.45:
                    bt_b, bt_a = build_biquad_bandpass(f3, bourdon_q, SR)
                    third, bourd_third_zi = lfilter(bt_b, bt_a, src, zi=bourd_third_zi)
                    voices = voices + third
                    n_voices += 1

            # high-Q bandpass on noise has low RMS; boost so the whistle sits with the wind.
            mix = mix + (voices / n_voices) * amp_eff * tone_level * 4.0

        mix = np.tanh(mix * drive)

        n_ch = outdata.shape[1]
        if spatial_mode and n_ch >= 2:
            gains = pan_gains(state.cur_position, n_ch, pan_floor)
            mix32 = mix.astype(np.float32)
            for c in range(n_ch):
                outdata[:, c] = mix32 * gains[c]
        else:
            mix32 = mix.astype(np.float32)
            for c in range(n_ch):
                outdata[:, c] = mix32

        # expose for TUI display
        state.cur_tilt = (math.log(max(60.0, f)) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))

    return callback


# ---- TUI -----------------------------------------------------------------

# (label, attr, min, max, step, fmt). Macro knobs at the top write through to fine
# knobs below (see State.apply_brightness / apply_whistle).
MACROS = [
    ("» brightness ", "brightness",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("» whistle    ", "whistle",      0.0,   1.0,   0.05, "{:>6.2f}   "),
]
KNOB_DEFS = MACROS + [
    ("low band  Fc", "low_fc",     20.0,  500.0,  10.0,  "{:>6.0f} Hz"),
    ("low band   Q", "low_q",       0.3,    5.0,   0.1,  "{:>6.2f}   "),
    ("high band Fc", "high_fc",  1000.0, 8000.0, 100.0,  "{:>6.0f} Hz"),
    ("high band  Q", "high_q",      1.0,   20.0,   0.5,  "{:>6.2f}   "),
    ("high band  G", "high_band_gain",0.0,  1.0,   0.05, "{:>6.2f}   "),
    ("mid Fc lo   ", "mid_fc_lo", 100.0,  800.0,  25.0,  "{:>6.0f} Hz"),
    ("mid Fc hi   ", "mid_fc_hi", 800.0, 5000.0, 100.0,  "{:>6.0f} Hz"),
    ("mid Q max   ", "mid_q_max",   0.0,   8.0,   0.25, "{:>6.2f}   "),
    ("gust depth  ", "gust_depth",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("gust tau    ", "gust_tau_s",   0.2,   8.0,   0.2,  "{:>6.1f} s "),
    ("Q drift     ", "q_drift_depth",0.0,   0.5,   0.02, "{:>6.2f}   "),
    ("Q drift tau ", "q_drift_tau_s",0.5,   8.0,   0.5,  "{:>6.1f} s "),
    ("drive       ", "drive",        0.5,  10.0,   0.2,  "{:>6.2f}   "),
    ("tone level  ", "tone_level",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("bourdon Q   ", "bourdon_q",    2.0,  30.0,   1.0,  "{:>6.1f}   "),
    ("pan floor   ", "pan_floor",    0.0,   0.5,   0.02, "{:>6.2f}   "),
]
MACRO_ATTRS = {a for _, a, *_ in MACROS}

THIRD_LABELS = {0: "off", 1: "min", 2: "maj"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def tui_loop(stdscr, state: State, port: str, baud: int, fake: bool = False):
    import curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 10 Hz redraw

    sel = 0
    while True:
        stdscr.erase()
        with state.lock:
            u3, ug = state.use_3band, state.use_gust
            u5, tm = state.use_fifth, state.third_mode
            usp = state.spatial_mode
            cp = state.cur_position
            knob_vals = [getattr(state, a) for _, a, *_ in KNOB_DEFS]
            note = state.note
            bend = state.pitch_bend
            cf = state.cur_freq
            ca = state.cur_amp
            cc = state.last_cc
            tilt = getattr(state, "cur_tilt", 0.0)

        try:
            stdscr.addstr(0, 0, "─── theremin wind ──────────────────────────────")
            if fake:
                stdscr.addstr(1, 0, f"input: trackpad fake — {port}")
            else:
                stdscr.addstr(1, 0, f"midi: {port} @ {baud}")

            stdscr.addstr(3, 0, "features:  ")
            stdscr.addstr(f"[{'x' if u3 else ' '}] 3-band (3)   ")
            stdscr.addstr(f"[{'x' if ug else ' '}] gust (g)   ")
            stdscr.addstr(f"[{'x' if u5 else ' '}] fifth (5)   ")
            stdscr.addstr(f"third: {THIRD_LABELS[tm]} (t)   ")
            stdscr.addstr(f"[{'x' if usp else ' '}] spatial (s)")

            stdscr.addstr(5, 0, "knobs:  (» = macro: writes through to fine knobs below)")
            for i, ((lab, attr, _, _, _, fmt), v) in enumerate(zip(KNOB_DEFS, knob_vals)):
                marker = "▶" if i == sel else " "
                line = f" {marker} {lab}  {fmt.format(v)}"
                attrs = curses.A_BOLD if attr in MACRO_ATTRS else 0
                if i == sel:
                    attrs |= curses.A_REVERSE
                stdscr.addstr(6 + i, 0, line, attrs)

            row = 6 + len(KNOB_DEFS) + 1
            stdscr.addstr(row, 0, "live:")
            stdscr.addstr(row + 1, 2,
                          f"note={str(note) if note is not None else '--':>3}  "
                          f"bend={bend:>+6}  freq={cf:>6.1f} Hz")
            spatial_str = f"  pos={cp:.2f}" if usp else ""
            stdscr.addstr(row + 2, 2,
                          f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  "
                          f"last_cc={cc[0]}={cc[1]}" if cc else
                          f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  last_cc=--")

            stdscr.addstr(row + 4, 0,
                          "↑↓ select knob   ←→ adjust   3/g/5/t/s toggle   q quit")
        except curses.error:
            pass  # window too small; skip this frame
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break
        if key == -1:
            continue
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            break

        with state.lock:
            if key == ord("3"):
                state.use_3band = not state.use_3band
            elif key == ord("g"):
                state.use_gust = not state.use_gust
            elif key == ord("5"):
                state.use_fifth = not state.use_fifth
            elif key == ord("t"):
                state.third_mode = (state.third_mode + 1) % 3
            elif key == ord("s"):
                state.spatial_mode = not state.spatial_mode
                # re-derive freq/position from current MIDI state under the new mode
                state._recompute_freq()
            elif key in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_LEFT, ord("h"), curses.KEY_RIGHT, ord("l")):
                _, attr, lo, hi, step, _ = KNOB_DEFS[sel]
                delta = -step if key in (curses.KEY_LEFT, ord("h")) else step
                setattr(state, attr, _clamp(getattr(state, attr) + delta, lo, hi))
                if attr == "brightness":
                    state.apply_brightness()
                elif attr == "whistle":
                    state.apply_whistle()


# ---- entry point ---------------------------------------------------------

def find_serial_port(requested: str | None) -> str:
    if requested:
        return requested
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not candidates:
        raise RuntimeError(
            "no /dev/ttyACM* or /dev/ttyUSB* found — is the OpenTheremin plugged in?"
        )
    return candidates[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="list serial + audio devices and exit")
    ap.add_argument("--serial", help="serial port (default: first /dev/ttyACM* or ttyUSB*)")
    ap.add_argument("--baud", type=int, default=BAUD,
                    help=f"baud rate (default {BAUD}; use 115200 if firmware is in USB-Hairless mode)")
    ap.add_argument("--audio", help="audio output device name substring")
    ap.add_argument("--debug", action="store_true",
                    help="print every MIDI message (disables TUI)")
    ap.add_argument("--no-tui", action="store_true",
                    help="don't open the curses TUI; just play with current knobs")
    ap.add_argument("--no-3band", action="store_true",
                    help="start with single-bandpass synth (toggle live with '3')")
    ap.add_argument("--no-gust", action="store_true",
                    help="start with gust LFO off (toggle live with 'g')")
    ap.add_argument("--spatial", action="store_true",
                    help="start in spatial mode: pitch antenna pans the wind L↔R "
                         "(toggle live with 's'); wind pitch becomes a fixed knob")
    ap.add_argument("--fake", action="store_true",
                    help="trackpad fake mode (no theremin needed): touchpad X = freq, Y = volume")
    ap.add_argument("--trackpad-dev",
                    help="path to /dev/input/eventN for the touchpad (default: autodetect)")
    ap.add_argument("--grab", action="store_true",
                    help="(fake mode) grab the touchpad exclusively so it doesn't move the cursor")
    args = ap.parse_args()

    if args.list:
        print("Serial ports:")
        for p in sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")):
            print(f"  {p}")
        print("\nAudio outputs:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print(f"  [{i}] {d['name']}  ({d['hostapi']})")
        return

    if args.fake:
        try:
            import evdev  # noqa: F401
        except ImportError:
            sys.exit("--fake needs the 'evdev' package: uv pip install --python .venv/bin/python evdev")
        try:
            if args.trackpad_dev:
                import evdev as _ev
                tpad = _ev.InputDevice(args.trackpad_dev)
            else:
                tpad = find_touchpad()
                if tpad is None:
                    sys.exit(
                        "no touchpad found. Either pass --trackpad-dev /dev/input/eventN, "
                        "or join the 'input' group: sudo usermod -aG input $USER  (then re-login)"
                    )
        except PermissionError:
            sys.exit(
                f"permission denied opening touchpad. Join the 'input' group:\n"
                f"  sudo usermod -aG input $USER   (then log out / log back in)"
            )
        port = f"{tpad.path}  [{tpad.name}]"
    else:
        port = find_serial_port(args.serial)

    if args.audio:
        sd.default.device = (None, args.audio)

    state = State()
    state.use_3band = not args.no_3band
    state.use_gust = not args.no_gust
    state.spatial_mode = args.spatial

    cb = make_audio_callback(state)

    if args.fake:
        input_thread = threading.Thread(
            target=trackpad_loop, args=(state, tpad, args.grab), daemon=True
        )
    else:
        input_thread = threading.Thread(
            target=serial_loop, args=(state, port, args.baud, args.debug), daemon=True
        )
    input_thread.start()

    use_tui = not (args.no_tui or args.debug)

    with sd.OutputStream(
        samplerate=SR, blocksize=BLOCK, channels=2,
        dtype="float32", callback=cb, latency="low",
    ):
        if use_tui:
            import curses
            try:
                curses.wrapper(tui_loop, state, port, args.baud, args.fake)
            except KeyboardInterrupt:
                pass
        else:
            print(f"[synth] pink={state.use_pink} 3band={state.use_3band} gust={state.use_gust}")
            print("[main] ctrl-c to stop. play your theremin.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    print("\n[main] bye")


if __name__ == "__main__":
    main()
