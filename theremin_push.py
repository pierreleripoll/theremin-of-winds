"""Push live theremin readings to the kiosk dashboard API.

This file is meant to run INSIDE the theremin process (copy it into the synth
repo next to dmx.py). It is a thin, fire-and-forget add-on that mirrors the DMX
thread pattern exactly: snapshot the two antenna values once under State.lock,
then POST them from this dedicated thread so the real-time audio callback is
never blocked or slowed. Every network error is swallowed - a missing or slow
kiosk must never affect the instrument; the dashboard just falls back to its own
simulation.

Uses only the stdlib (urllib), so it adds NO dependency to the theremin repo.

Wire it up in theremin_wind.py alongside the DMX thread, e.g.:

    import threading
    from theremin_push import push_loop
    ...
    if args.dashboard:
        threading.Thread(
            target=push_loop,
            args=(state, args.dashboard_url, stop),
            daemon=True,
        ).start()

where `stop` is a threading.Event set on shutdown, and `args.dashboard_url`
defaults to "http://localhost:8000/theremin-live".

The two values map straight onto the dashboard contract:
    right antenna -> frequency  = state.cur_freq    (smoothed pitch, Hz)
    left  antenna -> volume     = state.sound_level (gated output amplitude, 0..1)
"""
import json
import sys
import time
import urllib.error
import urllib.request

# Push rate. ~10 Hz is plenty to drive the charts smoothly and is negligible
# next to the audio/DMX work; the dashboard interpolates between frames anyway.
PUSH_HZ = 10.0
# Treat anything below this as silence (matches the API's own PLAY_EPS).
SOUND_EPS = 0.02


def _post(url, payload, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()  # drain so the connection can be reused/closed cleanly


def push_loop(state, url="http://localhost:8000/theremin-live", stop=None, debug=False):
    period = 1.0 / PUSH_HZ
    print(f"[dash] pushing live readings to {url} at {PUSH_HZ:g} Hz")
    while stop is None or not stop.is_set():
        t0 = time.monotonic()
        # Snapshot under the lock, exactly like the audio callback / DMX thread.
        with state.lock:
            freq = state.cur_freq
            vol = state.sound_level
            fan = bool(state.dmx_on)
        payload = {
            "frequency": float(freq),
            "volume": max(0.0, min(1.0, float(vol))),
            "fan_on": fan and vol > SOUND_EPS,
        }
        try:
            _post(url, payload, timeout=0.4)
        except (urllib.error.URLError, OSError) as e:
            # Kiosk down / unreachable: keep the instrument running regardless.
            if debug:
                print(f"[dash] push failed: {e}", file=sys.stderr)
        # Pace the loop without drifting; stop.wait so shutdown is immediate.
        dt = period - (time.monotonic() - t0)
        if dt > 0:
            if stop is not None:
                stop.wait(dt)
            else:
                time.sleep(dt)
