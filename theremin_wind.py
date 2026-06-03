#!/usr/bin/env python3
"""
Real-time wind-noise synth driven by an OpenTheremin V4 (MIDI firmware).

The OpenTheremin V4 with the MrDham/Vincent Dhamelincourt MIDI firmware sends
raw MIDI bytes over USB serial @ 31250 baud (it is NOT a USB-MIDI device).
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
import sys
import threading
import time

import sounddevice as sd

from audio import make_audio_callback
from autoplay import autoplay_loop
from config import BAUD, BLOCK, DMX_CHANNEL, DMX_MIN_LEVEL, DMX_ON_LEVEL, SR
from dmx import dmx_loop, find_dmx_port
from midi import find_serial_port, list_serial_ports, serial_loop
from state import State


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
                    help=f"DMX channel driving the fan/dimmer (default {DMX_CHANNEL})")
    ap.add_argument("--dmx-mode", choices=("switch", "dim"), default="switch",
                    help="switch = full-on while there's sound (dimmer channel in switch mode); "
                         "dim = fan speed tracks wind intensity (dimmer channel in gradation mode)")
    ap.add_argument("--dmx-min", type=int, default=DMX_MIN_LEVEL,
                    help=f"dim mode: lowest level that still spins the fan (default {DMX_MIN_LEVEL})")
    args = ap.parse_args()

    if args.list:
        print("Serial ports:")
        for p in list_serial_ports():
            print(f"  {p}")
        print("\nAudio outputs:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] > 0:
                print(f"  [{i}] {d['name']}  ({d['hostapi']})")
        return

    if args.autoplay:
        port = "autoplay (simulated, no hardware)"
    elif args.fake:
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
        port = find_serial_port(args.serial)

    if args.audio:
        sd.default.device = (None, args.audio)

    state = State()
    state.use_3band = not args.no_3band
    state.use_gust = not args.no_gust
    state.organ_mode = args.organ
    state.spatial_mode = args.spatial
    state.autoplay_on = args.autoplay

    cb = make_audio_callback(state)

    # Input source. --autoplay needs no hardware: the autoplay thread (started below,
    # always running but gated by state.autoplay_on) is the only input.
    if args.autoplay:
        input_thread = None
    elif args.fake:
        input_thread = threading.Thread(
            target=trackpad_loop, args=(state, tpad, args.grab), daemon=True
        )
    else:
        input_thread = threading.Thread(
            target=serial_loop, args=(state, port, args.baud, args.debug), daemon=True
        )
    if input_thread is not None:
        input_thread.start()

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
            sys.exit(str(e))
        state.dmx_available = True
        state.dmx_on = True
        dmx_thread = threading.Thread(
            target=dmx_loop,
            args=(state, dmx_port, args.dmx_channel, DMX_ON_LEVEL, dmx_stop, args.debug),
            kwargs={"proportional": args.dmx_mode == "dim", "min_level": args.dmx_min},
            daemon=True,
        )
        dmx_thread.start()

    use_tui = not (args.no_tui or args.debug)

    with sd.OutputStream(
        samplerate=SR, blocksize=BLOCK, channels=2,
        dtype="float32", callback=cb, latency="low",
    ):
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
    # blackout the fan and let the DMX thread close its port before we exit.
    dmx_stop.set()
    if dmx_thread is not None:
        dmx_thread.join(timeout=0.5)
    print("\n[main] bye")


if __name__ == "__main__":
    main()
