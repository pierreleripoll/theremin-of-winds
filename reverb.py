"""Stereo Freeverb (Schroeder-Moorer) reverb on the synth output bus.

A wet/dry reverb applied to the synth's stereo signal after saturation. It adds
NO latency to the dry path: the dry signal passes straight through the callback
and the reverb only adds a decaying tail on top (the comb/allpass delays affect
the wet signal only). It is built as a shared stereo bus so a future microphone
(the sung voice) can be summed into the same input and share one tail.

Efficiency note (why not a plain lfilter): a Freeverb comb is an order-~1200
recursive filter, and scipy's lfilter is O(order x frames) -- far too slow for
real time. But a comb is really a long delay line (read O(frames)) with a cheap
one-pole damping lowpass (order 1) in its feedback path. Because every delay D is
at least as long as the processing sub-block, no sample within a sub-block ever
feeds back on itself, so the whole sub-block vectorizes. Each block is split into
chunks no longer than the SHORTEST delay (the smallest allpass, ~245 samples at
48 kHz), so this holds for the short allpass lines too. Cost per block is a
handful of gather/scatter ops plus 16 order-1 lfilters -- negligible next to the
organ's additive stack.
"""
import numpy as np
from scipy.signal import lfilter

# Classic Freeverb tunings, in samples at 44.1 kHz; scaled to the run SR at init.
_COMB = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617]
_ALLPASS = [556, 441, 341, 225]
_STEREOSPREAD = 23          # right-bank delays are offset by this -> decorrelated L/R
_ALLPASS_FB = 0.5
_FIXED_GAIN = 0.015         # input gain into the comb bank (Freeverb's "fixedgain")
_ROOM_SCALE, _ROOM_OFFSET = 0.28, 0.7   # comb feedback = room*scale + offset (0.7..0.98)
_DAMP_SCALE = 0.4           # damping one-pole coef = damping*scale (0..0.4)
_WET_MAKEUP = 3.0           # makeup so a fully-wet mix sits near the dry's loudness


class _Comb:
    """Feedback comb with a one-pole lowpass ("damping") in the loop. Output is
    the delayed signal; the lowpassed output is fed back into the line."""

    def __init__(self, delay: int):
        self.buf = np.zeros(delay)
        self.pos = 0
        self.fs_zi = np.zeros(1)  # damping lowpass state, carried across blocks

    def process(self, x: np.ndarray, feedback: float, damp1: float) -> np.ndarray:
        m = x.shape[0]
        D = self.buf.shape[0]
        idx = (self.pos + np.arange(m)) % D
        out = self.buf[idx].copy()                       # delayed signal (the comb output)
        fs, self.fs_zi = lfilter([1.0 - damp1], [1.0, -damp1], out, zi=self.fs_zi)
        self.buf[idx] = x + feedback * fs
        self.pos = (self.pos + m) % D
        return out


class _Allpass:
    """Schroeder allpass (Freeverb's, with a fixed feedback of 0.5)."""

    def __init__(self, delay: int):
        self.buf = np.zeros(delay)
        self.pos = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        m = x.shape[0]
        D = self.buf.shape[0]
        idx = (self.pos + np.arange(m)) % D
        bufout = self.buf[idx].copy()
        self.buf[idx] = x + _ALLPASS_FB * bufout
        self.pos = (self.pos + m) % D
        return -x + bufout


class StereoReverb:
    """8-comb / 4-allpass stereo Freeverb. process() returns the WET signal only
    (no dry mixed in); the caller does the wet/dry blend so the same bus can serve
    several dry sources."""

    def __init__(self, sr: int):
        scale = sr / 44100.0

        def sc(d: int) -> int:
            return max(1, int(round(d * scale)))

        self.combs_l = [_Comb(sc(d)) for d in _COMB]
        self.combs_r = [_Comb(sc(d + _STEREOSPREAD)) for d in _COMB]
        self.aps_l = [_Allpass(sc(d)) for d in _ALLPASS]
        self.aps_r = [_Allpass(sc(d + _STEREOSPREAD)) for d in _ALLPASS]
        # chunk no longer than the shortest delay, so no line feeds back on itself
        # inside a sub-block (lets the whole sub-block vectorize).
        all_delays = [c.buf.shape[0] for c in self.combs_l + self.combs_r]
        all_delays += [a.buf.shape[0] for a in self.aps_l + self.aps_r]
        self.max_sub = max(1, min(all_delays))

    def process(self, dry: np.ndarray, room: float, damping: float) -> np.ndarray:
        """dry: (frames, 2). Returns wet (frames, 2)."""
        frames = dry.shape[0]
        feedback = room * _ROOM_SCALE + _ROOM_OFFSET
        damp1 = max(0.0, min(1.0, damping)) * _DAMP_SCALE
        inp = (dry[:, 0] + dry[:, 1]) * _FIXED_GAIN

        out_l = np.empty(frames)
        out_r = np.empty(frames)
        start = 0
        while start < frames:
            end = min(start + self.max_sub, frames)
            xs = inp[start:end]
            acc_l = np.zeros(end - start)
            acc_r = np.zeros(end - start)
            for c in self.combs_l:
                acc_l += c.process(xs, feedback, damp1)
            for c in self.combs_r:
                acc_r += c.process(xs, feedback, damp1)
            for ap in self.aps_l:
                acc_l = ap.process(acc_l)
            for ap in self.aps_r:
                acc_r = ap.process(acc_r)
            out_l[start:end] = acc_l
            out_r[start:end] = acc_r
            start = end

        return np.stack([out_l, out_r], axis=1) * _WET_MAKEUP
