"""MIDI-over-serial parsing and the serial reader thread.

The OpenTheremin V4 MIDI firmware sends raw MIDI bytes over USB serial at
31250 baud (it is NOT a USB-MIDI device). This module reads that stream
inline and dispatches into State.
"""
import glob
import sys
import time

import serial

from state import State


class MidiSerialReader:
    """Streaming MIDI byte parser. Handles running status; ignores sysex
    and 0xF8+ real-time bytes."""

    # status nibble -> number of data bytes
    DATA_LEN = {
        0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2,
        0xC0: 1, 0xD0: 1, 0xE0: 2,
    }

    def __init__(self, state: State, debug: bool):
        self.state = state
        self.debug = debug
        self.status = 0
        self.data: list[int] = []
        self.in_sysex = False

    def _dispatch(self, status: int, data: list[int]):
        kind = status & 0xF0
        ch = status & 0x0F
        if self.debug:
            print(f"[midi] ch{ch+1:>2} {kind:#04x} {data}")
        with self.state.lock:
            if kind == 0x90 and data[1] > 0:
                self.state.note_on(data[0], data[1])
            elif kind == 0x80 or (kind == 0x90 and data[1] == 0):
                self.state.note_off(data[0])
            elif kind == 0xE0:
                self.state.pitch_wheel(data[0], data[1])
            elif kind == 0xB0:
                self.state.cc(data[0], data[1])
            # 0xA0 (poly AT), 0xC0 (PC), 0xD0 (chan AT) ignored

    def feed(self, byte: int):
        # System real-time: interleavable, ignore
        if 0xF8 <= byte <= 0xFF:
            return

        # Sysex framing
        if byte == 0xF0:
            self.in_sysex = True
            return
        if byte == 0xF7:
            self.in_sysex = False
            return
        if self.in_sysex:
            return

        # System common (0xF1..0xF6) — clears running status, ignore
        if 0xF1 <= byte <= 0xF6:
            self.status = 0
            self.data = []
            return

        if byte & 0x80:
            # Channel status byte
            self.status = byte
            self.data = []
            return

        # Data byte. Need a valid running status.
        if not self.status:
            return
        self.data.append(byte)
        need = self.DATA_LEN.get(self.status & 0xF0, 0)
        if need and len(self.data) >= need:
            d = self.data[:need]
            self.data = []
            self._dispatch(self.status, d)


def serial_loop(state: State, port: str, baud: int, debug: bool):
    print(f"[midi] opening serial: {port} @ {baud}")
    while True:
        try:
            with serial.Serial(port, baud, timeout=0.05) as s:
                print("[midi] reading…")
                reader = MidiSerialReader(state, debug)
                while True:
                    chunk = s.read(64)
                    for b in chunk:
                        reader.feed(b)
        except serial.SerialException as e:
            print(f"[midi] serial error: {e}; retry in 2s", file=sys.stderr)
            time.sleep(2.0)


def find_serial_port(requested: str | None) -> str:
    if requested:
        return requested
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if not candidates:
        raise RuntimeError(
            "no /dev/ttyACM* or /dev/ttyUSB* found — is the OpenTheremin plugged in?"
        )
    return candidates[0]
