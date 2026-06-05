#!/usr/bin/env python3
"""Pick a warm light colour live with the trackpad, on the Colorbeam 7 (DMX addr 1).

Move your finger on the trackpad:
  left  <-> right : green amount  (left = pure red, right = amber/yellow)
  down  <-> up    : white amount  (down = saturated, up = warm white added)
Red is fixed at 255 and blue at 0 (blue/white is what made it pink). CH8 master
is held full so brightness stays constant while you choose the hue.

Lift your finger -> it prints the chosen RGBW as text. Adjust again as many times
as you like; each lift prints a new candidate. Ctrl-C to quit -- the LAST printed
line is the colour to keep.

  .venv/bin/python color_picker.py
"""
import sys
import time

import serial

from config import DMX_BAUD, DMX_FRAME_HZ
from dmx import _packet, find_dmx_port
from trackpad import open_touchpad

LIGHT_CH = 1          # R at DMX address 1, then G, B, W
G_MAX = 255           # full green travel (red -> yellow)
W_MAX = 200           # white travel (saturated -> opened)


def main():
    from evdev import ecodes

    dev = open_touchpad(None)
    ax = dev.absinfo(ecodes.ABS_X)
    ay = dev.absinfo(ecodes.ABS_Y)
    x_span = max(1, ax.max - ax.min)
    y_span = max(1, ay.max - ay.min)

    port = find_dmx_port(None)
    s = serial.Serial(port, DMX_BAUD, timeout=0.1)

    cur_x = (ax.min + ax.max) / 2
    cur_y = (ay.min + ay.max) / 2
    touching = False
    last = (255, 0, 0, 0)
    last_print = 0.0

    def frame_for(g, w):
        fr = bytearray(512)
        fr[LIGHT_CH - 1] = 255   # R
        fr[LIGHT_CH + 0] = g     # G
        fr[LIGHT_CH + 1] = 0     # B
        fr[LIGHT_CH + 2] = w     # W
        fr[7] = 255              # CH8 master (needed in 8-channel mode)
        return fr

    print(f"[picker] DMX {port}, trackpad {dev.path}")
    print("  gauche/droite = vert (rouge->ambre)   bas/haut = blanc")
    print("  leve le doigt = couleur validee (texte)   Ctrl-C = quitter\n")
    s.write(_packet(frame_for(0, 0)))  # start at pure red

    dev.grab()
    try:
        for ev in dev.read_loop():
            if ev.type == ecodes.EV_ABS:
                if ev.code == ecodes.ABS_X:
                    cur_x = ev.value
                elif ev.code == ecodes.ABS_Y:
                    cur_y = ev.value
            elif ev.type == ecodes.EV_KEY and ev.code == ecodes.BTN_TOUCH:
                touching = bool(ev.value)
                if not touching:
                    r, g, b, w = last
                    print(f"\r>>> COULEUR VALIDEE: R={r} G={g} B={b} W={w}    "
                          f"(CH1-4 = {r},{g},{b},{w})            ")
            elif ev.type == ecodes.EV_SYN and touching:
                xn = (cur_x - ax.min) / x_span
                yn = 1.0 - (cur_y - ay.min) / y_span
                g = int(max(0.0, min(1.0, xn)) * G_MAX)
                w = int(max(0.0, min(1.0, yn)) * W_MAX)
                last = (255, g, 0, w)
                s.write(_packet(frame_for(g, w)))
                now = time.time()
                if now - last_print > 0.08:
                    sys.stdout.write(f"\r    R=255 G={g:>3} B=0 W={w:>3}   ")
                    sys.stdout.flush()
                    last_print = now
    except KeyboardInterrupt:
        pass
    finally:
        try:
            dev.ungrab()
        except OSError:
            pass
        s.write(_packet(bytearray(512)))
        s.close()
        r, g, b, w = last
        print(f"\n\n[picker] derniere couleur: R={r} G={g} B={b} W={w}  -> on garde celle-la")


if __name__ == "__main__":
    main()
