#!/usr/bin/env python3
"""
Real-time wind-noise synth driven by an OpenTheremin V4.

By default it reads the custom value-stream firmware (OpenThereminV4/): ASCII
"P<pitch> V<vol> M<mute>" lines over USB serial @ 115200, where pitch/vol are the
firmware's linearized readings -- continuous, high-resolution control. With --midi
it reads the legacy MrDham MIDI firmware instead (raw MIDI @ 31250, not USB-MIDI).
Either way it drives a small real-time wind synth (band-passed white + low-passed
brown noise): pitch antenna -> frequency ("speed"), volume antenna -> amplitude
("intensity").

Usage:
  python theremin_wind.py --list                  # list serial + audio devices
  python theremin_wind.py                         # value-stream firmware (default)
  python theremin_wind.py --debug                 # print every reading, no TUI
  python theremin_wind.py --midi                  # legacy MIDI firmware
  python theremin_wind.py --serial /dev/ttyUSB0
"""
import argparse
import sys
import threading
import time

import sounddevice as sd

from audio import make_audio_callback
from autoplay import autoplay_loop
from config import (
    BAUD, BLOCK, DMX_CHANNEL, DMX_LIGHT_CHANNEL, DMX_LIGHT_COLOR, DMX_LIGHT_FLOOR,
    DMX_LIGHT_FULL_AT, DMX_LIGHT_PEAK, DMX_LIGHT_RISE_S, DMX_LIGHT_FALL_S,
    DMX_MIN_LEVEL, DMX_ON_LEVEL, DMX_STAGES, SR, STREAM_BAUD,
)
from dmx import dmx_loop, find_dmx_port
from midi import find_serial_port, list_serial_ports, serial_loop
from stream import stream_loop
from presets import load_preset
from state import State
from theremin_push import push_loop


def _resolve_device(substr: str, kind: str) -> int:
    """Resolve a device-name substring to a device index for 'input' or 'output'.
    Exits with the list of candidates if nothing matches -- clearer than PortAudio's
    raw error. Note: under PipeWire the interface is usually reached via the 'default'
    / 'pipewire' device rather than its own name, so prefer leaving this unset and
    selecting the card as the system default."""
    key = "max_input_channels" if kind == "input" else "max_output_channels"
    cands = [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d[key] > 0]
    hits = [(i, n) for i, n in cands if substr.lower() in n.lower()]
    if not hits:
        avail = "\n".join(f"  [{i}] {n}" for i, n in cands)
        sys.exit(f"no {kind} device matching {substr!r}. Available {kind}s:\n{avail}")
    return hits[0][0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="list serial + audio devices and exit")
    ap.add_argument("--serial", help="serial port (default: first /dev/ttyACM* or ttyUSB*)")
    ap.add_argument("--baud", type=int, default=None,
                    help=f"baud rate (default {STREAM_BAUD} for the value stream, {BAUD} for --midi)")
    ap.add_argument("--midi", action="store_true",
                    help="read the legacy MrDham MIDI firmware (raw MIDI @ 31250) instead of the "
                         "value stream; only needed if the OpenTheremin is reflashed back to MIDI")
    ap.add_argument("--audio", help="audio output device name substring")
    ap.add_argument("--mic", action="store_true",
                    help="capture a sung-voice mic from the sound-card input and run it "
                         "through the same reverb as the wind (opens a full-duplex stream)")
    ap.add_argument("--mic-device",
                    help="input device name substring for --mic (default: system default input)")
    ap.add_argument("--debug", action="store_true",
                    help="print every MIDI message (disables TUI)")
    ap.add_argument("--no-tui", action="store_true",
                    help="don't open the curses TUI; just play with current knobs")
    ap.add_argument("--no-3band", action="store_true",
                    help="start with single-bandpass synth (toggle live with '3')")
    ap.add_argument("--no-gust", action="store_true",
                    help="start with gust LFO off (toggle live with 'g')")
    ap.add_argument("--organ", action="store_true",
                    help="start in pipe-organ mode: additive harmonic voice + tremulant "
                         "instead of the wind voice (toggle live with 'o')")
    ap.add_argument("--spatial", action="store_true",
                    help="start in spatial mode: pitch antenna pans the wind L↔R "
                         "(toggle live with 's'); wind pitch becomes a fixed knob")
    ap.add_argument("--fake", action="store_true",
                    help="trackpad fake mode (no theremin needed): touchpad X = freq, Y = volume")
    ap.add_argument("--autoplay", action="store_true",
                    help="demo mode (no theremin needed): a background thread plays the synth "
                         "for you with slow random wind gestures (toggle live with 'a')")
    ap.add_argument("--maxautoplay", action="store_true",
                    help="like --autoplay but with the volume pinned full the whole time, so "
                         "only the pitch moves (toggle live with 'A')")
    ap.add_argument("--mute", action="store_true",
                    help="emit silence to the speakers while still driving the dashboard "
                         "(combine with --autoplay --dashboard for a signal-only kiosk; toggle 'm')")
    ap.add_argument("--trackpad-dev",
                    help="path to /dev/input/eventN for the touchpad (default: autodetect)")
    ap.add_argument("--grab", action="store_true",
                    help="(fake mode) grab the touchpad exclusively so it doesn't move the cursor")
    ap.add_argument("--dmx", action="store_true",
                    help="drive a DMX fan via an Enttec DMX USB Pro: full-on while the synth "
                         "makes sound, off when idle (toggle live with 'd')")
    ap.add_argument("--dmx-port",
                    help="Enttec serial port (default: autodetect the DMX USB Pro)")
    ap.add_argument("--dmx-channel", type=int, default=DMX_CHANNEL,
                    help=f"dimmer base address; fan outlets are offsets from it (default {DMX_CHANNEL})")
    ap.add_argument("--dmx-mode", choices=("switch", "dim"), default="switch",
                    help="switch = full-on while there's sound (dimmer channel in switch mode); "
                         "dim = fan speed tracks wind intensity (dimmer channel in gradation mode)")
    ap.add_argument("--dmx-min", type=int, default=DMX_MIN_LEVEL,
                    help=f"dim mode: lowest level that still spins the fan (default {DMX_MIN_LEVEL})")
    ap.add_argument("--no-dmx-light", action="store_false", dest="dmx_light",
                    help="don't drive the Colorbeam RGBW projector that breathes with the wind "
                         f"(on by default with --dmx, at DMX channel {DMX_LIGHT_CHANNEL})")
    ap.add_argument("--dmx-light-channel", type=int, default=DMX_LIGHT_CHANNEL,
                    help=f"projector base address (R; G/B/W follow) (default {DMX_LIGHT_CHANNEL})")
    ap.add_argument("--dashboard", action="store_true",
                    help="push live readings to the kiosk dashboard API")
    ap.add_argument("--dashboard-url", default="http://localhost:8000/theremin-live",
                    help="dashboard ingest endpoint (default: local FastAPI bridge)")
    args = ap.parse_args()
    if args.baud is None:
        args.baud = BAUD if args.midi else STREAM_BAUD

    if args.list:
        print("Serial ports:")
        for p in list_serial_ports():
            print(f"  {p}")
        print("\nAudio outputs:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print(f"  [{i}] {d['name']}  ({d['hostapi']})")
        print("\nAudio inputs (for --mic-device):")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']}  ({d['hostapi']})")
        return

    sim = False  # True only when no theremin is present and we fall back to a
    # hardware-less autoplay demo (autoplay then drives everything).
    if args.fake:
        try:
            from trackpad import open_touchpad, trackpad_loop
            tpad = open_touchpad(args.trackpad_dev)
        except ImportError:
            sys.exit("--fake needs the 'evdev' package: uv pip install --python .venv/bin/python evdev")
        except PermissionError:
            sys.exit(
                "permission denied opening touchpad. Join the 'input' group:\n"
                "  sudo usermod -aG input $USER   (then log out / log back in)"
            )
        except FileNotFoundError:
            sys.exit(
                "no touchpad found. Either pass --trackpad-dev /dev/input/eventN, "
                "or join the 'input' group: sudo usermod -aG input $USER  (then re-login)"
            )
        port = f"{tpad.path}  [{tpad.name}]"
    else:
        # Open a real theremin even with --autoplay/--maxautoplay: autoplay is a
        # background gesture generator gated by state.autoplay_on, so the live
        # theremin must stay wired up -- otherwise toggling autoplay off ('a')
        # leaves nothing driving the synth. Fall back to a hardware-less demo only
        # when no theremin is found and autoplay was requested.
        try:
            port = find_serial_port(args.serial)
        except RuntimeError as e:
            if args.autoplay or args.maxautoplay:
                port, sim = "autoplay (simulated, no hardware)", True
            else:
                sys.exit(str(e))

    # Resolve optional device substrings to explicit indices (None = system default).
    in_dev = _resolve_device(args.mic_device, "input") if (args.mic and args.mic_device) else None
    out_dev = _resolve_device(args.audio, "output") if args.audio else None

    state = State()
    # Restore the saved knobs/toggles (preset.json) if present, then let explicit
    # CLI flags force a toggle on top of the saved tuning.
    load_preset(state)
    if args.no_3band:
        state.use_3band = False
    if args.no_gust:
        state.use_gust = False
    if args.organ:
        state.organ_mode = True
    if args.spatial:
        state.spatial_mode = True
    state.autoplay_max = args.maxautoplay
    state.autoplay_on = args.autoplay or args.maxautoplay
    state.muted = args.mute

    cb = make_audio_callback(state, with_input=args.mic)

    # Input source. A real theremin is wired up whenever one is present (even in
    # autoplay) so toggling autoplay off hands control back to it; `sim` is set only
    # when no theremin was found and we fell back to a hardware-less autoplay demo.
    if args.fake:
        input_thread = threading.Thread(
            target=trackpad_loop, args=(state, tpad, args.grab), daemon=True
        )
    elif sim:
        input_thread = None  # no theremin; autoplay is the only input
    else:
        reader = serial_loop if args.midi else stream_loop
        input_thread = threading.Thread(
            target=reader, args=(state, port, args.baud, args.debug), daemon=True
        )
    if input_thread is not None:
        input_thread.start()

    # Rest-corner calibration only makes sense with a real theremin streaming MIDI
    # notes: --fake / --autoplay drive pitch through fake_xy (no note), so there is
    # no resting corner to sample. Started here, it auto-calibrates a couple seconds
    # after launch and holds the synth silent until it knows the rest point.
    calib_stop = threading.Event()
    if input_thread is not None and not args.fake:
        from calibrate import calibrate_loop
        threading.Thread(
            target=calibrate_loop, args=(state, calib_stop), daemon=True
        ).start()

    # Autoplay thread is always started; it only drives the synth while
    # state.autoplay_on is set, so the TUI can toggle demo mode live with 'a'.
    autoplay_stop = threading.Event()
    autoplay_thread = threading.Thread(
        target=autoplay_loop, args=(state, autoplay_stop), daemon=True
    )
    autoplay_thread.start()

    dmx_stop = threading.Event()
    dmx_thread = None
    if args.dmx:
        try:
            dmx_port = find_dmx_port(args.dmx_port)
        except RuntimeError as e:
            # Don't take the whole synth down because a DMX cable popped out --
            # for the exhibition the wind must keep playing. Warn and run without
            # the fan; the TUI just won't show the 'dmx fan' toggle.
            print(f"[dmx] {e} -- continuing without DMX fan", file=sys.stderr)
            dmx_port = None
        if dmx_port is not None:
            state.dmx_available = True
            state.dmx_on = True
            light = None
            if args.dmx_light:
                light = {
                    "base": args.dmx_light_channel,
                    "color": DMX_LIGHT_COLOR,
                    "floor": DMX_LIGHT_FLOOR, "peak": DMX_LIGHT_PEAK,
                    "full_at": DMX_LIGHT_FULL_AT,
                    "rise_s": DMX_LIGHT_RISE_S, "fall_s": DMX_LIGHT_FALL_S,
                }
            dmx_thread = threading.Thread(
                target=dmx_loop,
                args=(state, dmx_port, args.dmx_channel, DMX_ON_LEVEL, dmx_stop, args.debug),
                kwargs={"proportional": args.dmx_mode == "dim", "min_level": args.dmx_min,
                        "stages": DMX_STAGES, "light": light},
                daemon=True,
            )
            dmx_thread.start()

    push_stop = threading.Event()
    if args.dashboard:
        threading.Thread(
            target=push_loop,
            args=(state, args.dashboard_url, push_stop),
            kwargs={"debug": args.debug},
            daemon=True,
        ).start()

    use_tui = not (args.no_tui or args.debug)

    # --mic opens a full-duplex stream (one mono input channel + stereo out) so the
    # sung voice can share the reverb. Round-trip latency is one block in + one out
    # (~2 x BLOCK / SR); lower BLOCK in config.py if the singer needs tighter monitoring.
    if args.mic:
        stream = sd.Stream(
            samplerate=SR, blocksize=BLOCK, channels=(1, 2), device=(in_dev, out_dev),
            dtype="float32", callback=cb, latency="low",
        )
    else:
        stream = sd.OutputStream(
            samplerate=SR, blocksize=BLOCK, channels=2, device=out_dev,
            dtype="float32", callback=cb, latency="low",
        )
    with stream:
        if use_tui:
            import curses
            import os
            import tempfile
            from tui import tui_loop

            # The MIDI thread and the audio callback (and PortAudio/ALSA beneath it)
            # write diagnostics to stdout/stderr — e.g. "[audio] <status>" on every
            # xrun. While curses owns the terminal those writes scroll it underneath
            # the display, and since curses can't see the scroll the top row appears
            # to duplicate. Curses draws to the tty via fd 1, so we leave fd 1 alone
            # and send stdout (Python) plus stderr (fd 2, incl. C-level spam) to a
            # log file for the lifetime of the TUI, then restore them.
            log_path = os.path.join(tempfile.gettempdir(), "theremin_wind.log")
            logf = open(log_path, "a", buffering=1)
            saved_out, saved_err = sys.stdout, sys.stderr
            saved_err_fd = os.dup(2)
            sys.stdout = sys.stderr = logf
            os.dup2(logf.fileno(), 2)
            try:
                curses.wrapper(tui_loop, state, port, args.baud, args.fake)
            except KeyboardInterrupt:
                pass
            finally:
                os.dup2(saved_err_fd, 2)
                os.close(saved_err_fd)
                sys.stdout, sys.stderr = saved_out, saved_err
                logf.close()
            print(f"[main] TUI diagnostics were logged to {log_path}")
        else:
            print(f"[synth] 3band={state.use_3band} gust={state.use_gust}")
            print("[main] ctrl-c to stop. play your theremin.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass

    autoplay_stop.set()
    push_stop.set()
    calib_stop.set()
    # blackout the fan and let the DMX thread close its port before we exit.
    dmx_stop.set()
    if dmx_thread is not None:
        dmx_thread.join(timeout=0.5)
    print("\n[main] bye")


if __name__ == "__main__":
    main()
