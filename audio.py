"""Real-time audio synthesis.

Signal flow per block:
    pink noise (Paul Kellet 6-pole IIR) →
        either three parallel RBJ bandpass biquads with per-band gain envelopes
        (the "spectral tilt") or one bandpass + one lowpass (legacy single-band)
    → multiply by amp × gust LFO
    → optional bourdon voices (narrow bandpass at root + 5th + 3rd)
    → tanh saturation (drive knob)
    → optional stereo pan (spatial mode)

Filter coefficients are recomputed per block (cheap), so any knob change takes
effect within ~10 ms.
"""
import math
import sys

import numpy as np
from scipy.signal import iirfilter, lfilter, lfilter_zi

from config import (
    BLOCK, FREQ_HI, FREQ_LO, RUMBLE_CUTOFF, SMOOTH_MS, SR,
)
from state import State

# Paul Kellet's refined pink noise filter (6 parallel one-poles + white passthrough).
# Sounds smoother than Voss-McCartney; cheap when run via lfilter per pole.
PINK_POLES = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616]
PINK_GAINS = [0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980]
PINK_DIRECT = 0.5362
PINK_SCALE = 0.11


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

    def lfo_step(prev: float, alpha: float, depth: float) -> tuple[float, float]:
        """One block of a unit-variance random-walk LFO. Returns (new_state, mod)
        where mod ∈ [1-depth, 1+depth] is meant to multiply amp/Q. The norm
        rescales because a one-pole on unit-variance noise has stationary
        stddev sqrt(alpha/(2-alpha))."""
        new_state = (1.0 - alpha) * prev + alpha * rng.standard_normal()
        norm = math.sqrt((2.0 - alpha) / alpha)
        return new_state, 1.0 + depth * math.tanh(new_state * norm * 0.7)

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

        if use_gust:
            gust_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(gust_tau_s, 0.05))
            gust_state, gust_mod = lfo_step(gust_state, gust_alpha, gust_depth)
        else:
            gust_mod = 1.0
        amp_eff = amp * gust_mod

        if q_drift_depth > 0.0:
            q_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(q_drift_tau_s, 0.05))
            q_drift_state, q_mod = lfo_step(q_drift_state, q_alpha, q_drift_depth)
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

        # Bourdon: narrow bandpass on the same pink noise at root + optional 5th/3rd.
        # Pitched-wind whistle (like air across a bottle), not a sine pad. Tracks the
        # theremin in parallel — the singer-and-shadowing-monk effect of medieval organum.
        if tone_level > 0.0:
            def voice(freq: float, zi):
                b, a = build_biquad_bandpass(freq, bourdon_q, SR)
                return lfilter(b, a, src, zi=zi)

            # f is pre-clamped to <= SR * 0.45; root is always safe to filter.
            voices, bourd_root_zi = voice(f, bourd_root_zi)
            n_voices = 1
            if use_fifth and f * 1.5 < SR * 0.45:
                fifth, bourd_fifth_zi = voice(f * 1.5, bourd_fifth_zi)
                voices = voices + fifth
                n_voices += 1
            if third_mode > 0:
                ratio = 1.2 if third_mode == 1 else 1.25  # 6:5 minor, 5:4 major
                if f * ratio < SR * 0.45:
                    third, bourd_third_zi = voice(f * ratio, bourd_third_zi)
                    voices = voices + third
                    n_voices += 1

            # high-Q bandpass on noise has low RMS; boost so the whistle sits with the wind.
            mix = mix + (voices / n_voices) * amp_eff * tone_level * 4.0

        mix32 = np.tanh(mix * drive).astype(np.float32)

        n_ch = outdata.shape[1]
        if spatial_mode and n_ch >= 2:
            gains = np.asarray(pan_gains(state.cur_position, n_ch, pan_floor),
                               dtype=np.float32)
            outdata[:] = mix32[:, None] * gains
        else:
            outdata[:] = mix32[:, None]

        # expose for TUI display
        state.cur_tilt = (math.log(max(60.0, f)) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))

    return callback
