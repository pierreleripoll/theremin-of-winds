"""MIDI-over-serial parsing and the serial reader thread.

The OpenTheremin V4 MIDI firmware sends raw MIDI bytes over USB serial at
31250 baud (it is NOT a USB-MIDI device). This module reads that stream
inline and dispatches into State.
"""
import sys
import time

import serial
import serial.tools.list_ports

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
            self.state.msg_count += 1
            if self.state.autoplay_on:
                # autoplay owns the synth; ignore the theremin (which may be
                # streaming its rest corner) so the two don't fight the targets.
                return
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


# The OpenTheremin V4 talks over a CH340 USB-serial bridge (WCH, USB vendor
# 0x1a86). Prefer it over any other USB-serial device: audio interfaces like
# the MOTU M2 also expose a serial control port that sorts ahead of ttyUSB0
# and would otherwise be picked first, leaving the synth listening to silence.
THEREMIN_VIDS = (0x1A86,)

# Devices that share the USB-serial bus but are positively NOT the theremin, so
# auto-selection must never fall back to them. The Enttec DMX USB Pro (FTDI
# 0403:6001) is the one that bites: if the theremin is unplugged, picking the
# Enttec makes the MIDI reader silently "read" the DMX port and parse nothing,
# which looks exactly like a dead theremin.
NON_THEREMIN_VID_PIDS = ((0x0403, 0x6001),)


def list_serial_ports() -> list[str]:
    """Candidate USB-serial ports, theremin (CH340) first.

    Filters to real USB devices (those report a vendor id), dropping the
    motherboard's legacy /dev/ttyS* ports. Lists everything (incl. the Enttec)
    so --list stays useful for diagnostics; auto-selection is stricter.
    """
    ports = [p for p in serial.tools.list_ports.comports() if p.vid is not None]
    ports.sort(key=lambda p: (p.vid not in THEREMIN_VIDS, p.device))
    return [p.device for p in ports]


def find_serial_port(requested: str | None) -> str:
    if requested:
        return requested
    ports = [p for p in serial.tools.list_ports.comports() if p.vid is not None]
    # Never auto-grab a port we know is not a theremin (the Enttec DMX), so an
    # absent theremin raises a clear error instead of leaving MIDI reading the
    # DMX port. CH340 (THEREMIN_VIDS) still sorts ahead of any other adapter.
    candidates = [p for p in ports if (p.vid, p.pid) not in NON_THEREMIN_VID_PIDS]
    candidates.sort(key=lambda p: (p.vid not in THEREMIN_VIDS, p.device))
    if not candidates:
        raise RuntimeError(
            "no OpenTheremin serial port found — is it plugged in? "
            "(the Enttec DMX port is ignored on purpose)"
        )
    return candidates[0].device
