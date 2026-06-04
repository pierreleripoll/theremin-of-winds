"""Continuous value-stream reader for the custom OpenTheremin V4 firmware.

The firmware in OpenThereminV4/ (build.h: SERIAL_PORT_MODE_STREAM) emits ASCII
lines "P<pitch> V<vol> M<mute>\\n" at ~200 Hz over USB serial @ 115200 baud, where
pitch/vol are the firmware's linearized readings and mute is the play/standby
button. This is the alternative to the MIDI firmware (midi.py): continuous,
high-resolution control with no MIDI quantization. Port autodetection (the CH340
bridge) is shared with midi.py.
"""
import sys
import time

import serial

from midi import find_serial_port, list_serial_ports  # noqa: F401  (re-exported)
from state import State


class StreamReader:
    """Parses the firmware's "P<pitch> V<vol> M<mute>\\n" lines into State.

    Tolerant of partial reads (keeps a buffer across chunks) and of the odd
    garbled line (a line that doesn't parse is dropped, not fatal)."""

    def __init__(self, state: State, debug: bool):
        self.state = state
        self.debug = debug
        self.buf = b""

    def feed(self, chunk: bytes):
        self.buf += chunk
        # Keep the last (possibly partial) line in the buffer; parse complete ones.
        *lines, self.buf = self.buf.split(b"\n")
        for line in lines:
            self._parse(line)

    def _parse(self, line: bytes):
        try:
            text = line.decode("ascii", "ignore").strip()
        except Exception:
            return
        if not text:
            return
        pitch = vol = None
        playing = True
        for tok in text.split():
            tag, val = tok[:1], tok[1:]
            try:
                n = int(val)
            except ValueError:
                return  # malformed token -> drop the whole line
            if tag == "P":
                pitch = n
            elif tag == "V":
                vol = n
            elif tag == "M":
                playing = bool(n)
        if pitch is None or vol is None:
            return
        if self.debug:
            print(f"[stream] P={pitch} V={vol} M={int(playing)}")
        with self.state.lock:
            self.state.stream_input(pitch, vol, playing)


def stream_loop(state: State, port: str, baud: int, debug: bool):
    print(f"[stream] opening serial: {port} @ {baud}")
    while True:
        try:
            with serial.Serial(port, baud, timeout=0.05) as s:
                print("[stream] reading…")
                reader = StreamReader(state, debug)
                while True:
                    chunk = s.read(256)
                    if chunk:
                        reader.feed(chunk)
        except serial.SerialException as e:
            print(f"[stream] serial error: {e}; retry in 2s", file=sys.stderr)
            time.sleep(2.0)
