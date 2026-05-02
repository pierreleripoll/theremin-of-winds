"""Shared mutable state between the input thread, audio callback, and TUI.

Everything that crosses thread boundaries lives here, guarded by `State.lock`.
The audio callback snapshots all values it needs once per block under the lock,
then synthesizes without holding it.
"""
import threading

from config import (
    DRIVE, FREQ_HI, FREQ_LO, GUST_DEPTH, GUST_TAU_S, HIGH_BAND_GAIN,
    HIGH_FC, HIGH_Q, LOW_FC, LOW_Q, MID_FC_HI, MID_FC_LO, MID_Q_MAX,
    NOTE_HI, NOTE_LO, Q_DRIFT_DEPTH, Q_DRIFT_TAU_S,
)


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
        # macros: setting these writes through to the fine knobs above.
        # Defaults of 1.0 reproduce the historical "stormy" sound; 1.0 maps to the
        # same fine-knob values just assigned, so direct-set bypasses the setter.
        self._brightness = 1.0
        self._whistle = 1.0
        # exposed for the TUI; written by the audio callback each block.
        self.cur_tilt = 0.0

    @property
    def brightness(self) -> float:
        return self._brightness

    @brightness.setter
    def brightness(self, v: float):
        # 0 = dark/calm, 1 = bright/stormy. Maps to high_band_gain and drive.
        self._brightness = v
        self.high_band_gain = HIGH_BAND_GAIN * v
        self.drive = 0.5 + (DRIVE - 0.5) * v

    @property
    def whistle(self) -> float:
        return self._whistle

    @whistle.setter
    def whistle(self, v: float):
        # 0 = no resonance/howl, 1 = full whistle ramp. Maps to high_q and mid_q_max.
        self._whistle = v
        self.high_q = 1.0 + (HIGH_Q - 1.0) * v
        self.mid_q_max = MID_Q_MAX * v

    def recompute_freq(self):
        if self.note is None:
            return
        bend_semis = self.pitch_bend / 8192.0 * 2.0  # default ±2 semitones
        n = max(NOTE_LO, min(NOTE_HI, self.note + bend_semis))
        ratio = (n - NOTE_LO) / (NOTE_HI - NOTE_LO)
        self.target_freq = FREQ_LO * (FREQ_HI / FREQ_LO) ** ratio

    def note_on(self, note: int, vel: int):
        self.note = note
        self.recompute_freq()
        if self.target_amp == 0.0:
            self.target_amp = (vel / 127.0) ** 1.8

    def note_off(self, note: int):
        if self.note == note:
            self.note = None
            self.target_amp = 0.0

    def pitch_wheel(self, lsb: int, msb: int):
        self.pitch_bend = ((msb << 7) | lsb) - 8192
        self.recompute_freq()

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
