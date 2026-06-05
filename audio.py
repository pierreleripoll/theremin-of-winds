"""Real-time audio synthesis.

Signal flow per block:
    pink noise (Paul Kellet 6-pole IIR) →
        either three parallel RBJ bandpass biquads with per-band gain envelopes
        (the "spectral tilt") or one bandpass + one lowpass (legacy single-band)
    → multiply by amp × gust LFO
    → optional bourdon voices (narrow bandpass at root + 5th + 3rd)
    → tanh saturation (drive knob)
    → optional stereo pan (spatial mode)

The noise source is generated as two independent realisations (left/right),
correlated by rho = 1 - stereo_width, and run through the SAME filters, so the wind
widens into an enveloping stereo field without comb-filtering (the "stereo width"
knob). The organ's tonal partials are summed mono (centred); only the breath spreads.

Organ mode layers a pipe-organ drone on top of the wind voice (it does not replace
it): an additive harmonic stack (sine partials at multiples of a deep root note)
blended with pink noise resonating at the same harmonics (the "air in the metal
tube"). It is driven by the SAME gust LFO as the wind, which swells, brightens, and
sharpens it together with the audible wind — so it sounds like wind resonating a
building, not a separate organ. The sweep is compressed onto a narrow bass range.

Filter coefficients are recomputed per block (cheap), so any knob change takes
effect within ~10 ms.
"""
import math
import sys

import numpy as np
from scipy.signal import iirfilter, lfilter, lfilter_zi

from config import (
    AMP_EPS, AMP_RISE_S, BLOCK, FREQ_HI, FREQ_LO, IDLE_TIMEOUT_S,
    INERTIA_IDLE_FULL_S, INERTIA_MAX_ADD_S, REST_AMP_MARGIN, REST_NOTE_MARGIN,
    RUMBLE_CUTOFF, SMOOTH_MS, SR,
    STORM_DRIVE, STORM_GUST_DEPTH, STORM_GUST_TAU, STORM_HIGH, STORM_MIDQ,
    STORM_SOFT_BODY, STORM_SOFT_DARK,
)
from reverb import StereoReverb
from state import State

# Paul Kellet's refined pink noise filter (6 parallel one-poles + white passthrough).
# Sounds smoother than Voss-McCartney; cheap when run via lfilter per pole.
PINK_POLES = [0.99886, 0.99332, 0.96900, 0.86650, 0.55000, -0.7616]
PINK_GAINS = [0.0555179, 0.0750759, 0.1538520, 0.3104856, 0.5329522, -0.0168980]
PINK_DIRECT = 0.5362
PINK_SCALE = 0.11

# Pipe organ "principal chorus": harmonic multiples of the played pitch and their
# raw levels (8' 4' 2-2/3' 2' 1-3/5' 1-1/3' 1'). The 7th harmonic is skipped — no
# classical organ rank sits at 7x. Levels are normalized to sum 1 at synth time so
# the worst-case aligned peak stays <= 1 before any gain.
ORGAN_MULTS = np.array([1, 2, 3, 4, 5, 6, 8], dtype=np.float64)
ORGAN_AMPS = np.array([1.00, 0.55, 0.30, 0.22, 0.13, 0.10, 0.07], dtype=np.float64)
ORGAN_GAIN = 0.85   # headroom so the bright partials aren't crushed by the tanh
BRIGHT_TAPER = 0.7  # how far organ_brightness can pull down the partials above the root
AIR_Q = 9.0         # resonance of the "air in the tube" bandpasses (higher = more whistly)
AIR_BOOST = 4.0     # high-Q bandpass on noise has low RMS; boost it to sit with the tone
ORGAN_WAVER = 0.006 # depth of the slow random pitch waver (natural wind detune)
ORGAN_WAVER_TAU = 1.2  # pitch waver correlation time (s)
# Wind coupling: a gust above/below the wind's mean also brightens and sharpens the
# organ (like rising air pressure overblowing a flue pipe), so it tracks the wind.
ORGAN_WIND_BRIGHT = 0.6  # how much a gust brightens the organ
ORGAN_WIND_PITCH = 0.01  # how much a gust sharpens the organ pitch
# Organ is a narrow bass drone, not a wide melodic voice: the theremin's full sweep
# is compressed onto a deep root note spanning only ORGAN_RANGE_SEMIS semitones.
ORGAN_BASE_HZ = 220.0     # lowest organ note at octave 0 (hand at the bottom of the sweep)
ORGAN_RANGE_SEMIS = 7.0   # how many semitones the whole sweep spans above the base


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
    rumble_b, rumble_a = iirfilter(
        2, RUMBLE_CUTOFF / (SR / 2.0), btype="low", ftype="butter"
    )
    # All filter states carry two columns (left/right noise) and are filtered along
    # axis 0, so the stereo decorrelation costs nothing structurally: width just sets
    # how independent the two noise columns are (see the noise source in the callback).
    # output-bus reverb (stereo Freeverb), applied after the drive saturation.
    reverb = StereoReverb(SR)

    rumble_zi = np.zeros((2, 2))
    bp_zi = np.zeros((2, 2))

    low_zi = np.zeros((2, 2))
    mid_zi = np.zeros((2, 2))
    high_zi = np.zeros((2, 2))

    pink_zis = [np.zeros((1, 2)) for _ in PINK_POLES]
    gust_state = 0.0
    q_drift_state = 0.0

    # zi persists across blocks so coef changes (player moves the theremin) don't click.
    bourd_root_zi = np.zeros((2, 2))
    bourd_fifth_zi = np.zeros((2, 2))
    bourd_third_zi = np.zeros((2, 2))

    # organ-mode oscillator + tremulant phase, carried across blocks like the zi above.
    organ_phase = 0.0
    trem_phase = 0.0
    organ_waver_state = 0.0  # slow random-walk pitch waver
    organ_swell = 0.0  # slow envelope: how present the organ is, built by sustained strong wind
    organ_air_zi = [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))]  # air bands (f0, 2f0, 3f0)

    tau_blocks = (SMOOTH_MS / 1000.0) * SR / BLOCK
    alpha_smooth = 1.0 - math.exp(-1.0 / max(tau_blocks, 1.0))
    # slower one-pole for amplitude RISES only (anti-blast slew, see callback).
    rise_blocks = AMP_RISE_S * SR / BLOCK
    alpha_amp_up = 1.0 - math.exp(-1.0 / max(rise_blocks, 1.0))

    # auto-wind presence gate (see config.py): the firmware stops sending MIDI when
    # no hand is near, so we fade the output out once messages stop arriving.
    presence = 0.0
    last_msg_count = 0
    blocks_since_msg = 0
    idle_secs = 0.0  # accumulated idle time -> restart inertia (see config.py)

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
        nonlocal organ_phase, trem_phase, organ_waver_state, organ_swell
        nonlocal presence, last_msg_count, blocks_since_msg, idle_secs
        if status:
            print(f"[audio] {status}", file=sys.stderr)

        with state.lock:
            tf, ta = state.target_freq, state.target_amp
            attack_s = state.attack_s
            release_s = state.release_s
            msg_count = state.msg_count
            note = state.note
            rest_amp, rest_note = state.rest_amp, state.rest_note
            calibrating = state.calibrating
            use_3band = state.use_3band
            use_gust = state.use_gust
            use_fifth = state.use_fifth
            third_mode = state.third_mode
            organ_mode = state.organ_mode
            organ_octave = state.organ_octave
            organ_brightness = state.organ_brightness
            organ_air, organ_wind = state.organ_air, state.organ_wind
            organ_level = state.organ_level
            organ_threshold = state.organ_threshold
            organ_rise_s, organ_fall_s = state.organ_rise_s, state.organ_fall_s
            trem_depth, trem_rate, trem_pitch = state.trem_depth, state.trem_rate, state.trem_pitch
            tone_level = state.tone_level
            bourdon_q = state.bourdon_q
            spatial_mode = state.spatial_mode
            stereo_width = state.stereo_width
            reverb_mix = state.reverb_mix
            reverb_room = state.reverb_room
            reverb_damping = state.reverb_damping
            muted = state.muted
            solo = set(state.solo)
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
            storm = state.storm

        # idle = no new MIDI messages for IDLE_TIMEOUT_S (firmware went quiet).
        fresh_msg = msg_count != last_msg_count
        if fresh_msg:
            last_msg_count = msg_count
            blocks_since_msg = 0
        else:
            blocks_since_msg += 1
        idle = blocks_since_msg >= IDLE_TIMEOUT_S * SR / BLOCK

        # Rest-corner gate: at rest (no hands) the instrument streams max volume +
        # lowest note -- the corner of the control space. Once calibrated, a reading
        # within a margin of that corner means nobody is playing, so we silence the
        # wind the same way the idle timeout does (the firmware doesn't go quiet at
        # rest, so the idle gate alone never catches this). `calibrating` holds it
        # silent while the rest point is sampled, including the first seconds after
        # launch -- which is what stops the synth blasting on boot.
        resting = (
            rest_amp is not None
            and ta >= rest_amp - REST_AMP_MARGIN
            and (rest_note is None or note is None or note <= rest_note + REST_NOTE_MARGIN)
        )
        gated_off = resting or calibrating

        state.cur_freq += (tf - state.cur_freq) * alpha_smooth
        # Asymmetric amplitude slew. The volume antenna reads "hand far = loud", so
        # pulling a hand off the instrument (or just standing away) spikes the target
        # toward full. Amplitude only rises slowly (alpha_amp_up) and only while fresh
        # MIDI keeps arriving AND we aren't gated off -- so a yanked hand can't reach
        # full, a stale loud target can't keep pushing the wind up across an input gap,
        # and the resting corner never charges cur_amp toward max behind the gate.
        # Falling (hand nearing the volume antenna = quieter) stays fast so the wind
        # can still be cut by hand.
        if ta > state.cur_amp:
            if fresh_msg and not gated_off:
                state.cur_amp += (ta - state.cur_amp) * alpha_amp_up
        else:
            state.cur_amp += (ta - state.cur_amp) * alpha_smooth
        state.cur_position += (tp - state.cur_position) * alpha_smooth

        if idle or gated_off:
            # nobody playing: fade out, and bank idle time as restart inertia.
            idle_secs = min(idle_secs + BLOCK / SR, INERTIA_IDLE_FULL_S)
            rel_tau_blocks = max(release_s, 0.05) * SR / BLOCK
            presence += (0.0 - presence) * (1.0 - math.exp(-1.0 / rel_tau_blocks))
            if presence < AMP_EPS:
                presence = 0.0  # truly silent
        else:
            # spin up from zero like a wind machine overcoming inertia: the longer it
            # sat idle, the slower the restart (ATTACK_S + up to INERTIA_MAX_ADD_S).
            # Once spun up (presence ~ full) the banked inertia is spent, so play is
            # fully responsive again until the next long idle.
            inertia = min(idle_secs / INERTIA_IDLE_FULL_S, 1.0)
            eff_attack_s = attack_s + INERTIA_MAX_ADD_S * inertia
            att_tau_blocks = max(eff_attack_s, 0.01) * SR / BLOCK
            presence += (1.0 - presence) * (1.0 - math.exp(-1.0 / att_tau_blocks))
            if presence >= 1.0 - AMP_EPS:
                idle_secs = 0.0

        f = max(60.0, min(SR * 0.45, state.cur_freq))
        amp = max(0.0, min(1.0, state.cur_amp)) * presence
        state.sound_level = float(amp)  # read by the DMX thread to gate the fan

        # Storm morph: let the played volume drive the wind's CHARACTER, not just its
        # level -- soft = poetic breeze, loud = violent storm. `m` is biased to the loud
        # end (amp^1.5) so soft play stays calm and the storm only hits near full volume;
        # `soft` grows as you play quieter (the muffled-breeze terms). At storm=0 every
        # effective value below collapses to the old expression.
        s = storm
        m = s * (amp ** 1.5)
        soft = s * (1.0 - amp)
        drive_eff = drive * (1.0 + STORM_DRIVE * m)

        # Two independent white-noise realisations for left/right, correlated by rho so
        # the two output channels can be decorrelated without ever phase-combing (the
        # failure of the old allpass widener): rho=1 (width 0) = identical columns ->
        # dual mono; rho=0 (width 1) = independent -> fully diffuse. A linear mix of
        # independent noises stays flat-spectrum at every width, in stereo AND summed to
        # mono (mono just loses up to 3 dB of level, never a notch). Both columns run
        # the SAME filters below (axis 0): identical timbre, independent grain.
        rho = 1.0 - max(0.0, min(1.0, stereo_width))
        nL = rng.standard_normal(frames).astype(np.float64)
        nR = rho * nL + math.sqrt(max(0.0, 1.0 - rho * rho)) * rng.standard_normal(frames)
        # Paul Kellet's 6-pole IIR + white passthrough, run on both columns at once.
        white = np.stack([nL, nR], axis=1) * 0.4   # (frames, 2)
        src = white * PINK_DIRECT
        for i, (pole, gain) in enumerate(zip(PINK_POLES, PINK_GAINS)):
            y, pink_zis[i] = lfilter([gain], [1.0, -pole], white, axis=0, zi=pink_zis[i])
            src = src + y
        src *= PINK_SCALE

        # Wind voice (always on). Organ mode layers a pipe-organ drone on top.
        # The gust LFO is the wind's slow breath; compute it whenever the wind OR the
        # organ needs it, so the organ can couple to the same gusting (see below).
        if use_gust or organ_mode:
            # storm: bigger and faster gusts when loud, gentle slow breath when soft.
            gust_depth_eff = min(0.95, gust_depth * (1.0 + STORM_GUST_DEPTH * m))
            gust_tau_eff = gust_tau_s * (1.0 - STORM_GUST_TAU * m)
            gust_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(gust_tau_eff, 0.05))
            gust_state, gust_mod = lfo_step(gust_state, gust_alpha, gust_depth_eff)
        else:
            gust_mod = 1.0
        amp_eff = amp * gust_mod if use_gust else amp

        if q_drift_depth > 0.0:
            q_alpha = 1.0 - math.exp(-(BLOCK / SR) / max(q_drift_tau_s, 0.05))
            q_drift_state, q_mod = lfo_step(q_drift_state, q_alpha, q_drift_depth)
        else:
            q_mod = 1.0

        # Each audio layer is accumulated into `comp` so the TUI's solo flags can
        # isolate any subset of them (see the solo mix near the tanh below).
        comp: dict[str, np.ndarray] = {}

        if use_3band:
            tilt = (math.log(f) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))
            tilt = max(0.0, min(1.0, tilt))

            mid_fc = mid_fc_lo * (mid_fc_hi / mid_fc_lo) ** tilt
            mid_Q = (1.2 + mid_q_max * (amp ** 0.6) * (1.0 + STORM_MIDQ * m)) * q_mod

            # rebuild biquads each block — cheap and lets knobs change live
            low_b, low_a = build_biquad_bandpass(low_fc, low_q, SR)
            mid_b, mid_a = build_biquad_bandpass(mid_fc, mid_Q, SR)
            high_b, high_a = build_biquad_bandpass(high_fc, high_q, SR)

            low, low_zi = lfilter(low_b, low_a, src, axis=0, zi=low_zi)
            mid, mid_zi = lfilter(mid_b, mid_a, src, axis=0, zi=mid_zi)
            high, high_zi = lfilter(high_b, high_a, src, axis=0, zi=high_zi)

            # muffle the mid AND high bands for the soft poetic breeze. The high band
            # alone is pitch-gated (tilt**3 ~ 0 at low/mid pitch), so darkening it does
            # nothing audible when playing soft and low -- the mid band is the body you
            # actually hear there, so the darkening has to reach it too.
            g_dark = 1.0 - STORM_SOFT_DARK * soft
            g_low = ((1.0 - tilt) * 0.6 + 0.3) * (1.0 + STORM_SOFT_BODY * soft)
            g_mid = 0.7 * g_dark
            g_high = (tilt ** 3) * high_band_gain * (1.0 + STORM_HIGH * m) * g_dark

            comp["low"] = low * g_low * 4.0 * amp_eff
            comp["mid"] = mid * g_mid * 1.5 * amp_eff
            comp["high"] = high * g_high * 1.5 * amp_eff
        else:
            Q = (1.2 + (mid_q_max + 1.0) * (amp ** 0.6) * (1.0 + STORM_MIDQ * m)) * q_mod
            b, a = build_biquad_bandpass(f, Q, SR)
            bp, bp_zi = lfilter(b, a, src, axis=0, zi=bp_zi)
            rumble, rumble_zi = lfilter(rumble_b, rumble_a, src, axis=0, zi=rumble_zi)
            # single-band wind isn't split into grave/medium/aigu; expose the whole
            # voice under "low" so band-solo still surfaces it (band solo is a 3band thing).
            comp["low"] = (bp * 1.5 * 0.75 + rumble * 4.0 * 0.55) * amp_eff

        # Bourdon: narrow bandpass on the same pink noise at root + optional 5th/3rd.
        # Pitched-wind whistle (like air across a bottle), not a sine pad. Tracks the
        # theremin in parallel — the singer-and-shadowing-monk effect of medieval organum.
        if tone_level > 0.0:
            def voice(freq: float, zi):
                b, a = build_biquad_bandpass(freq, bourdon_q, SR)
                return lfilter(b, a, src, axis=0, zi=zi)

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
            comp["bourdon"] = (voices / n_voices) * amp_eff * tone_level * 4.0

        # Organ voice: a pipe-organ drone driven by the same wind you hear as noise.
        if organ_mode:
            # gust_dev = how hard the wind is blowing right now vs its mean. It is the
            # wind's slow breath (shared with the noise voice above), and we let it
            # swell, brighten, and sharpen the organ together with the audible wind so
            # the two feel like one system — wind resonating a building. organ_wind
            # sets how tightly the organ tracks it.
            gust_dev = gust_mod - 1.0
            waver_alpha = 1.0 - math.exp(-(frames / SR) / ORGAN_WAVER_TAU)
            organ_waver_state, waver_mod = lfo_step(organ_waver_state, waver_alpha, ORGAN_WAVER)
            trem_phase = math.fmod(trem_phase + 2.0 * math.pi * trem_rate * frames / SR,
                                   2.0 * math.pi)
            trem = math.sin(trem_phase)

            # swell with inertia: the organ only builds while the wind sits above
            # organ_threshold, rising over organ_rise_s and lingering over organ_fall_s.
            # Strong wind must hold for seconds before the organ is fully present, and
            # it fades out gradually rather than tracking the wind instantly.
            demand = max(0.0, (amp - organ_threshold) / max(1e-6, 1.0 - organ_threshold))
            swell_tau = organ_rise_s if demand > organ_swell else organ_fall_s
            swell_alpha = 1.0 - math.exp(-(frames / SR) / max(swell_tau, 0.05))
            organ_swell += (demand - organ_swell) * swell_alpha

            organ_amp_eff = (amp * organ_level * organ_swell
                             * (1.0 + organ_wind * gust_dev) * (1.0 + trem_depth * trem))
            bright = min(1.0, max(0.0, organ_brightness + ORGAN_WIND_BRIGHT * organ_wind * gust_dev))

            # compress the theremin's full sweep onto a narrow bass range: recover the
            # hand position (0..1) from the pitch, then span only ORGAN_RANGE_SEMIS
            # semitones above a deep base note. waver + vibrato + the wind's overblow
            # (gust_dev) detune it slightly for a living, wind-driven pitch.
            ratio = (math.log(f) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))
            ratio = max(0.0, min(1.0, ratio))
            f0 = ORGAN_BASE_HZ * (2.0 ** organ_octave) * (2.0 ** (ratio * ORGAN_RANGE_SEMIS / 12.0))
            f0 = f0 * waver_mod * (1.0 + trem_pitch * trem) * (1.0 + ORGAN_WIND_PITCH * organ_wind * gust_dev)
            f0 = max(20.0, min(SR * 0.45, f0))

            # principal chorus partials at multiples of the root.
            keep = (ORGAN_MULTS * f0) < (SR * 0.45)
            keep[0] = True  # the fundamental always plays
            mults = ORGAN_MULTS[keep]
            amps = ORGAN_AMPS[keep].copy()
            # brightness (wind-coupled) pulls down the partials above the root;
            # renormalize so the overall level holds as it moves.
            amps[mults >= 2.0] *= (1.0 - BRIGHT_TAPER) + BRIGHT_TAPER * bright
            amps /= max(amps.sum(), 1e-9)

            # One fundamental phase, carried across blocks; harmonic k is exactly
            # k * phase, so the partials stay phase-locked and seamless at block joins.
            n = np.arange(frames, dtype=np.float64)
            phase = organ_phase + (2.0 * math.pi * f0 / SR) * n
            stack = (amps[:, None] * np.sin(mults[:, None] * phase[None, :])).sum(axis=0)
            organ_phase = math.fmod(organ_phase + 2.0 * math.pi * f0 * frames / SR,
                                    2.0 * math.pi)

            # "air in the metal tube": pink noise resonating at the tube's first few
            # harmonics, so the breath is pitched (whistling through the pipe), not
            # broadband hiss. This is what makes it sound like wind, not a synth organ.
            air = 0.0
            if organ_air > 0.0:
                for j, (k, w) in enumerate(((1.0, 1.0), (2.0, 0.6), (3.0, 0.4))):
                    fk = f0 * k
                    if fk < SR * 0.45:
                        ab, aa = build_biquad_bandpass(fk, AIR_Q, SR)
                        band, organ_air_zi[j] = lfilter(ab, aa, src, axis=0, zi=organ_air_zi[j])
                        air = air + band * w

            # stack is the deterministic tonal partials -> mono (centred). air is the
            # stereo noise breath. stack[:, None] broadcasts the tone onto both columns.
            comp["organ"] = (stack[:, None] * ORGAN_GAIN + air * AIR_BOOST * organ_air) * organ_amp_eff

        # Solo: with any solo flag set, mix only those layers so the player can hear
        # one in isolation (grave/medium/aigu/bourdon/orgue). Empty set = mix all.
        keys = comp.keys() & solo if solo else comp.keys()
        mix = np.zeros((frames, 2), dtype=np.float64)
        for k in keys:
            mix = mix + comp[k]

        state.cur_tilt = (math.log(max(60.0, f)) - math.log(FREQ_LO)) / (math.log(FREQ_HI) - math.log(FREQ_LO))

        # Dashboard-only / mute: sound_level + cur_freq are already written above, so
        # the kiosk still animates; just emit silence to the speakers.
        if muted:
            outdata[:] = 0.0
            return

        # Saturate the stereo bus, then add the reverb tail on top of the saturated
        # (dry) signal -- a clean tail rather than the drive grinding the reverb. The
        # dry passes through untouched, so the reverb adds no latency; only mix > 0
        # spends any CPU. This is the shared bus a future sung-voice mic would join.
        sat = np.tanh(mix * drive_eff)
        if reverb_mix > 0.0:
            wet = reverb.process(sat, reverb_room, reverb_damping)
            sat = sat * (1.0 - reverb_mix) + wet * reverb_mix

        n_ch = outdata.shape[1]
        # sat is a decorrelated stereo pair. On a stereo device, emit it directly.
        # Spatial mode is a hand-controlled point source, not a diffuse field, so it
        # folds to mono and pans; mono/surround devices fold down and replicate.
        if spatial_mode and n_ch >= 2:
            mono = sat.mean(axis=1).astype(np.float32)
            gains = np.asarray(pan_gains(state.cur_position, n_ch, pan_floor),
                               dtype=np.float32)
            outdata[:] = mono[:, None] * gains
        elif n_ch == 2:
            outdata[:] = sat.astype(np.float32)
        else:
            outdata[:] = sat.mean(axis=1).astype(np.float32)[:, None]

    return callback
