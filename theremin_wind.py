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
from config import BAUD, BLOCK, SR
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
    ap.add_argument("--spatial", action="store_true",
                    help="start in spatial mode: pitch antenna pans the wind L↔R "
                         "(toggle live with 's'); wind pitch becomes a fixed knob")
    ap.add_argument("--fake", action="store_true",
                    help="trackpad fake mode (no theremin needed): touchpad X = freq, Y = volume")
    ap.add_argument("--trackpad-dev",
                    help="path to /dev/input/eventN for the touchpad (default: autodetect)")
    ap.add_argument("--grab", action="store_true",
                    help="(fake mode) grab the touchpad exclusively so it doesn't move the cursor")
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
        port = find_serial_port(args.serial)

    if args.audio:
        sd.default.device = (None, args.audio)

    state = State()
    state.use_3band = not args.no_3band
    state.use_gust = not args.no_gust
    state.spatial_mode = args.spatial

    cb = make_audio_callback(state)

    if args.fake:
        input_thread = threading.Thread(
            target=trackpad_loop, args=(state, tpad, args.grab), daemon=True
        )
    else:
        input_thread = threading.Thread(
            target=serial_loop, args=(state, port, args.baud, args.debug), daemon=True
        )
    input_thread.start()

    use_tui = not (args.no_tui or args.debug)

    with sd.OutputStream(
        samplerate=SR, blocksize=BLOCK, channels=2,
        dtype="float32", callback=cb, latency="low",
    ):
        if use_tui:
            import curses
            from tui import tui_loop
            try:
                curses.wrapper(tui_loop, state, port, args.baud, args.fake)
            except KeyboardInterrupt:
                pass
        else:
            print(f"[synth] 3band={state.use_3band} gust={state.use_gust}")
            print("[main] ctrl-c to stop. play your theremin.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    print("\n[main] bye")


if __name__ == "__main__":
    main()
