"""Auto-play (demo) mode: make the synth play itself, no theremin needed.

A background thread wanders the pitch and volume the way a hand would on the
instrument -- easing between random "gesture" waypoints with occasional lulls --
and drives them through State.fake_xy, which also bumps msg_count so the auto-wind
presence gate (audio.py) keeps the sound alive. Gated by State.autoplay_on so the
TUI can toggle it live ('a'); started on with --autoplay.
"""
import random

from config import (
    AUTOPLAY_AMP_HI, AUTOPLAY_AMP_LO, AUTOPLAY_MAX_GESTURE_S,
    AUTOPLAY_MIN_GESTURE_S, AUTOPLAY_PITCH_STEP, AUTOPLAY_RATE_HZ,
    AUTOPLAY_REST_AMP, AUTOPLAY_REST_PROB,
)
from state import State


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def autoplay_loop(state: State, stop_event, rng: random.Random | None = None):
    rng = rng or random.Random()
    dt = 1.0 / AUTOPLAY_RATE_HZ
    # the simulated hand position: x = pitch (0..1), y = volume (0..1). We glide
    # from the gesture start (x0, y0) to its target (tx, ty) over `glide` ∈ [0,1];
    # glide >= 1.0 means the gesture is done, so pick the next waypoint.
    x0 = rng.random()
    y0 = AUTOPLAY_AMP_LO
    tx, ty = x0, y0
    glide = 1.0
    step = 0.0

    while not stop_event.is_set():
        if not state.autoplay_on:
            stop_event.wait(0.1)
            continue

        if glide >= 1.0:
            x0, y0 = tx, ty
            tx = _clamp01(tx + rng.uniform(-AUTOPLAY_PITCH_STEP, AUTOPLAY_PITCH_STEP))
            if rng.random() < AUTOPLAY_REST_PROB:
                ty = rng.uniform(0.0, AUTOPLAY_REST_AMP)  # a lull: wind eases down
            else:
                ty = rng.uniform(AUTOPLAY_AMP_LO, AUTOPLAY_AMP_HI)
            step = dt / rng.uniform(AUTOPLAY_MIN_GESTURE_S, AUTOPLAY_MAX_GESTURE_S)
            glide = 0.0

        glide = min(1.0, glide + step)
        e = glide * glide * (3.0 - 2.0 * glide)  # smoothstep ease in/out
        cx = x0 + (tx - x0) * e
        cy = y0 + (ty - y0) * e
        with state.lock:
            state.fake_xy(cx, cy)
        stop_event.wait(dt)
