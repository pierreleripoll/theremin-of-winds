# Task: push live readings to the CERN-kiosk dashboard

This synth needs a thin add-on that streams its live values to the fake-CERN kiosk
installation, where a dashboard "island" on a (spoofed) `home.cern` project page animates
charts from them. The kiosk and its backend already exist; **all that remains is wiring the
push from this repo.**

- Sibling repo: `/mnt/data/Documents/Code/copy-cern` (the fake CERN homepage + dashboard).
- Backend already built there: a FastAPI bridge at `copy-cern/proxy/api/` that holds the
  latest frame and serves it to the dashboard over SSE. It exposes
  `POST /theremin-live` for this synth to push to. See `copy-cern/proxy/api/README.md`.
- A ready-made, dependency-free pusher is already written:
  **`copy-cern/proxy/api/theremin_push.py`** (stdlib `urllib` only). The job is mostly to
  copy it in and start it like the DMX thread.

## What to send

Two values, a few times a second (the pusher does this at ~10 Hz):

| field       | source                | meaning                          |
|-------------|-----------------------|----------------------------------|
| `frequency` | `state.cur_freq`      | right antenna -> pitch (Hz)      |
| `volume`    | `state.sound_level`   | left antenna -> wind (0..1)      |
| `fan_on`    | `state.dmx_on` & sound| optional; the fan/DMX state      |

The backend derives the rest (`playing` = volume > 0.02, `activity`, timestamp) and forces
`playing`/`fan_on` false if no push arrives for 2s. Nothing here needs to know the dashboard
contract beyond these fields.

## Steps

1. **Copy the pusher in:** `cp ../copy-cern/proxy/api/theremin_push.py .`
   (No new dependency — it uses stdlib `urllib`. Do not add `requests`.)

2. **Add CLI flags** in `theremin_wind.py`, next to the `--dmx` args:
   ```python
   ap.add_argument("--dashboard", action="store_true",
                   help="push live readings to the kiosk dashboard API")
   ap.add_argument("--dashboard-url", default="http://localhost:8000/theremin-live",
                   help="dashboard ingest endpoint (default: local FastAPI bridge)")
   ```

3. **Start it as a daemon thread**, mirroring the DMX block exactly (snapshot under
   `State.lock`, separate thread, never touches the audio callback). Add after the DMX
   thread is started:
   ```python
   from theremin_push import push_loop
   ...
   push_stop = threading.Event()
   if args.dashboard:
       threading.Thread(
           target=push_loop,
           args=(state, args.dashboard_url, push_stop),
           kwargs={"debug": args.debug},
           daemon=True,
       ).start()
   ```

4. **Stop it on shutdown**, next to `dmx_stop.set()`:
   ```python
   push_stop.set()
   ```
   (It's a daemon thread, so a join is optional; the pusher checks `stop` on every loop.)

## Threading discipline (do not break)

- The audio callback is real-time. The pusher must NEVER run in it and never hold
  `state.lock` for more than the one-line snapshot — exactly like `dmx_loop`.
- All network errors are already swallowed inside `push_loop`: a missing/slow kiosk must not
  affect the instrument. Keep it that way.

## Test

Run the backend in the sibling repo, then the synth pointed at it:

```sh
# terminal 1 - the bridge
cd ../copy-cern/proxy/api && pip install -r requirements.txt && python3 main.py

# terminal 2 - confirm pushes land
watch -n0.5 'curl -s localhost:8000/healthz'      # "live": true while you play

# terminal 3 - the synth (use --fake if no theremin is plugged in)
python3 theremin_wind.py --dashboard --fake
```

`healthz` should report `live: true` and a small `last_push_age_s` while sound is produced.
To see the charts move, open the dashboard from the kiosk proxy (see copy-cern's README).
