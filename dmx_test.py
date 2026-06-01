#!/usr/bin/env python3
"""Force one DMX channel to a fixed value to test the dimmer/fan wiring on its
own, with the synth out of the picture. Sends continuously until ctrl-c, then
blacks out the channel.

  .venv/bin/python dmx_test.py            # channel 50 -> 255 (full on)
  .venv/bin/python dmx_test.py 50 128     # channel 50 -> 128 (~switch threshold)
  .venv/bin/python dmx_test.py 50 0       # channel 50 -> 0   (off)
  .venv/bin/python dmx_test.py 0 255      # ALL 512 channels -> 255 (ignore addressing)

Channel 0 drives every channel at once: if the fan reacts to that but not to
channel 50, it is an addressing/outlet problem (wrong address or wrong Schuko).

Watch the [DMX] LED on the dimmer while this runs: it lights only when a valid
DMX signal arrives AND the unit is in Sla/DMX mode. LED off -> no DMX reaching
the dimmer (mode not Sla, or cable/terminator/Enttec). LED on -> DMX is arriving
and it is an address / channel-function / outlet issue.
"""
import sys
import time

import serial

from config import DMX_BAUD, DMX_FRAME_HZ
from dmx import _packet, find_dmx_port


def main():
    channel = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    value = int(sys.argv[2]) if len(sys.argv) > 2 else 255
    channel = max(0, min(512, channel))
    value = max(0, min(255, value))

    port = find_dmx_port(None)
    where = "ALL channels" if channel == 0 else f"channel {channel}"
    print(f"[dmx-test] {port}: {where} = {value}  (ctrl-c to stop)")

    frame = bytearray(512)
    if channel == 0:
        frame = bytearray([value]) * 512
    else:
        frame[channel - 1] = value
    with serial.Serial(port, DMX_BAUD, timeout=0.1) as s:
        try:
            while True:
                s.write(_packet(frame))
                time.sleep(1.0 / DMX_FRAME_HZ)
        except KeyboardInterrupt:
            pass
        s.write(_packet(bytearray(512)))  # blackout every channel on the way out
    print("\n[dmx-test] blackout, bye")


if __name__ == "__main__":
    main()
