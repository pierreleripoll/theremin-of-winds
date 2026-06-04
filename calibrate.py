"""Rest-position calibration for the OpenTheremin.

The instrument's resting posture (no hands near) is the corner of the control
space: loudest volume + lowest pitch (both antennas read "hand far"). The MIDI
firmware keeps streaming that corner even when nobody is there, so on its own the
synth boots into a full-blast low wind. We sample the corner -- once automatically
a couple of seconds after launch (step back while it does) and again whenever the
operator presses 'c' -- and store it on State. The audio gate then treats any
reading within a small margin of the corner as "nobody playing" and stays silent.

Only started for real serial input: in --fake / --autoplay the pitch comes from
fake_xy (no MIDI note), so there is no rest corner to calibrate.
"""
import time

from config import CALIB_SAMPLE_S, CALIB_STARTUP_DELAY_S
from state import State


def _sample(state: State, dur: float) -> bool:
    """Watch the incoming readings for `dur` seconds and record the resting corner:
    the loudest volume and the lowest note seen. At rest both sit at their extremes,
    so max-amp / min-note recover the corner even through a little antenna jitter.

    Returns False (leaving rest_* untouched) if no MIDI arrived during the window --
    nothing is streaming, so the idle-timeout gate is the right fallback."""
    amps: list[float] = []
    notes: list[int] = []
    start = state.msg_count
    t0 = time.monotonic()
    while time.monotonic() - t0 < dur:
        with state.lock:
            amps.append(state.target_amp)
            if state.note is not None:
                notes.append(state.note)
        time.sleep(0.02)
    if state.msg_count == start or not amps:
        return False
    with state.lock:
        state.rest_amp = max(amps)
        state.rest_note = float(min(notes)) if notes else None
    return True


def calibrate_loop(state: State, stop):
    """Auto-calibrate once at startup, then re-sample whenever 'c' is pressed.

    `calibrating` holds the synth silent through every sample (including the first
    seconds after launch), so the instrument never blasts before it knows its rest
    point, and the sample is taken while the operator is stepping back."""
    # Hold the synth silent from boot through the first sample so it never blasts
    # before it knows its rest point. But if we launched into autoplay the sound is
    # intentional: don't silence it, and defer the first sample until autoplay is
    # switched off -- autoplay both masks the rest corner and suppresses the MIDI
    # the sample needs (see midi.py). The activity-timeout gate covers idle meanwhile.
    if not state.autoplay_on:
        with state.lock:
            state.calibrating = True
    if not stop.wait(CALIB_STARTUP_DELAY_S):
        while state.autoplay_on and not stop.is_set():
            stop.wait(0.05)
        if not stop.is_set():
            with state.lock:
                state.calibrating = True
            _sample(state, CALIB_SAMPLE_S)
    with state.lock:
        state.calibrating = False

    while not stop.wait(0.05):
        with state.lock:
            req = state.calibrate_request
            if req:
                state.calibrate_request = False
                state.calibrating = True
        if req:
            _sample(state, CALIB_SAMPLE_S)
            with state.lock:
                state.calibrating = False
