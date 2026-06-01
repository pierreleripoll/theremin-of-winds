"""DMX output to an Enttec DMX USB Pro: drive a fan when the synth makes sound.

The Enttec DMX USB Pro is NOT a raw DMX serializer — it speaks a labelled packet
protocol over its FTDI serial port and generates the DMX-512 timing itself. We
send "Output Only Send DMX Packet" (label 6) frames; the device keeps emitting the
last frame on the wire, so we just refresh it ~40x/s.

Packet: 0x7E, label, len_lsb, len_msb, <data...>, 0xE7
where data[0] is the DMX start code (0x00) and data[1..512] are channels 1..512.

Wiring here: a power dimmer addressed at channel 50 feeds the fan, driven by
`State.sound_level` (the gated output amplitude, written by the audio callback).
Two modes: "switch" drives the channel full-on while there's sound; "dim" maps
the wind's intensity to fan speed (floored at min_level so the motor keeps
spinning). A short hold bridges note-to-note gaps and a gentle ramp keeps the
dimmer from being slammed.
"""
import math
import sys
import time

import serial
import serial.tools.list_ports

from config import (
    DMX_BAUD, DMX_FRAME_HZ, DMX_FULL_AT, DMX_HOLD_S, DMX_RAMP_S, DMX_SOUND_EPS,
)
from state import State

# The Enttec DMX USB Pro is an FTDI device (VID 0x0403 / PID 0x6001) reporting
# product "DMX USB PRO". The OpenTheremin sits on a CH340 (VID 0x1a86), so there
# is no ambiguity between the two even when both are /dev/ttyUSB*.
ENTTEC_VID_PID = (0x0403, 0x6001)


def find_dmx_port(requested: str | None) -> str:
    if requested:
        return requested
    for p in serial.tools.list_ports.comports():
        if (p.vid, p.pid) == ENTTEC_VID_PID or (p.product and "DMX USB PRO" in p.product.upper()):
            return p.device
    raise RuntimeError(
        "no Enttec DMX USB Pro found — is it plugged in? (pass --dmx-port to force)"
    )


def _packet(channels: bytearray) -> bytes:
    """Wrap a 512-channel frame in an Enttec 'Send DMX' (label 6) packet."""
    data = b"\x00" + bytes(channels)  # DMX start code + 512 channels
    n = len(data)
    return bytes((0x7E, 6, n & 0xFF, (n >> 8) & 0xFF)) + data + b"\xE7"


def dmx_loop(state: State, port: str, channel: int, on_level: int, stop, debug: bool,
             proportional: bool = False, min_level: int = 0):
    channel = max(1, min(512, channel))
    on_level = max(0, min(255, on_level))
    min_level = max(0, min(on_level, min_level))
    mode = "dim" if proportional else "switch"
    print(f"[dmx] opening Enttec: {port} (channel {channel}, mode={mode}, "
          f"min={min_level}, on={on_level})")
    period = 1.0 / DMX_FRAME_HZ
    ramp_alpha = 1.0 - math.exp(-period / max(DMX_RAMP_S, 1e-3))
    frame = bytearray(512)  # channels 1..512 -> index 0..511
    cur = 0.0
    last_sound = -1e9

    while not stop.is_set():
        try:
            with serial.Serial(port, DMX_BAUD, timeout=0.1) as s:
                print("[dmx] sending…")
                while not stop.is_set():
                    now = time.monotonic()
                    level = state.sound_level
                    if level > DMX_SOUND_EPS:
                        last_sound = now
                    # hold the fan on briefly past the last sound so note-to-note
                    # gaps (and the release tail) don't chatter the dimmer.
                    on = state.dmx_on and (now - last_sound) < DMX_HOLD_S
                    if not on:
                        target = 0
                    elif proportional:
                        # map wind intensity to fan speed, floored at min_level so
                        # the motor keeps spinning instead of stalling/buzzing.
                        frac = max(0.0, min(1.0, level / max(DMX_FULL_AT, 1e-6)))
                        target = min_level + (on_level - min_level) * frac
                    else:
                        target = on_level
                    cur += (target - cur) * ramp_alpha
                    frame[channel - 1] = int(max(0, min(255, round(cur))))
                    if debug:
                        print(f"[dmx] ch{channel}={frame[channel - 1]:>3} on={on}")
                    s.write(_packet(frame))
                    time.sleep(period)
                # blackout on the way out so the fan doesn't keep spinning.
                frame[channel - 1] = 0
                try:
                    s.write(_packet(frame))
                except serial.SerialException:
                    pass
        except serial.SerialException as e:
            print(f"[dmx] serial error: {e}; retry in 2s", file=sys.stderr)
            stop.wait(2.0)
