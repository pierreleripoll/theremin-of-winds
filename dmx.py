"""DMX output to an Enttec DMX USB Pro: drive a fan when the synth makes sound.

The Enttec DMX USB Pro is NOT a raw DMX serializer — it speaks a labelled packet
protocol over its FTDI serial port and generates the DMX-512 timing itself. We
send "Output Only Send DMX Packet" (label 6) frames; the device keeps emitting the
last frame on the wire, so we just refresh it ~40x/s.

Packet: 0x7E, label, len_lsb, len_msb, <data...>, 0xE7
where data[0] is the DMX start code (0x00) and data[1..512] are channels 1..512.

Wiring here: a power dimmer at `base_channel` feeds curtain fans, and each
`stages` entry drives one outlet off `State.sound_level` (the gated output
amplitude written by the audio callback) with its own loudness threshold and
inertia. So a low-threshold "play" stage wakes one fan as soon as the synth is
played, while a high-threshold "storm" stage only adds a second fan after the
wind has been very loud for a while; each engages with inertia (a charge
integrates loudness) and keeps blowing through a brief complete stop (see the
DMX_STAGES notes in config). Two modes: "switch" drives an engaged channel
full-on; "dim" maps the wind's intensity to fan speed (floored at min_level so
the motor keeps spinning). A gentle ramp keeps the dimmer from being slammed.
"""
import math
import sys
import time

import serial
import serial.tools.list_ports

from config import DMX_BAUD, DMX_FRAME_HZ, DMX_FULL_AT, DMX_RAMP_S
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


def dmx_loop(state: State, port: str, base_channel: int, on_level: int, stop, debug: bool,
             proportional: bool = False, min_level: int = 0, stages=None, light=None):
    base_channel = max(1, min(512, base_channel))
    on_level = max(0, min(255, on_level))
    min_level = max(0, min(on_level, min_level))
    mode = "dim" if proportional else "switch"
    period = 1.0 / DMX_FRAME_HZ
    ramp_alpha = 1.0 - math.exp(-period / max(DMX_RAMP_S, 1e-3))

    # Optional RGBW light (Colorbeam) on the same DMX universe: its brightness
    # breathes with sound_level between a resting floor and a peak, smoothed with a
    # fast-ish swell and a slow ember-like fall (asymmetric one-pole). Fixed colour;
    # only the level moves. Held master (base+7) covers the fixture's 8-channel mode.
    lt = None
    if light:
        lb = max(1, min(512, light["base"]))
        lt = {
            "chans": [lb + i for i in range(4)],   # R, G, B, W
            "master": lb + 7,                       # 8-channel-mode master; harmless in 4-ch
            "color": light["color"],
            "floor": light["floor"], "peak": light["peak"],
            "full_at": max(light["full_at"], 1e-6),
            "rise": 1.0 - math.exp(-period / max(light["rise_s"], 1e-3)),
            "fall": 1.0 - math.exp(-period / max(light["fall_s"], 1e-3)),
            "bright": light["floor"],
        }

    # One independent integrator per outlet. Built once so charge/engaged/cur
    # persist across serial reconnects.
    st = []
    for label, outlet, loud_at, attack_s, decay_s, engage_at, disengage_at in (stages or []):
        st.append({
            "label": label,
            "ch": max(1, min(512, base_channel + outlet - 1)),
            "loud_at": loud_at,
            "attack": 1.0 - math.exp(-period / max(attack_s, 1e-3)),
            "decay": 1.0 - math.exp(-period / max(decay_s, 1e-3)),
            "engage_at": engage_at, "disengage_at": disengage_at,
            "charge": 0.0, "engaged": False, "cur": 0.0,
        })

    desc = ", ".join(f"{s['label']}->ch{s['ch']}@{s['loud_at']}" for s in st)
    if lt:
        desc += f", light->ch{lt['chans'][0]}-{lt['chans'][3]} {int(lt['floor']*100)}-{int(lt['peak']*100)}%"
    print(f"[dmx] opening Enttec: {port} (mode={mode}, min={min_level}, on={on_level}; {desc})")
    frame = bytearray(512)  # channels 1..512 -> index 0..511

    while not stop.is_set():
        try:
            with serial.Serial(port, DMX_BAUD, timeout=0.1) as s:
                print("[dmx] sending…")
                while not stop.is_set():
                    level = state.sound_level
                    enabled = state.dmx_on
                    for stg in st:
                        # Integrate loudness: rise (attack) while the wind clears
                        # this stage's threshold, decay slowly otherwise so a short
                        # complete stop doesn't drop the fan. Hysteresis on the
                        # engage/disengage thresholds avoids chatter at the boundary.
                        loud = level >= stg["loud_at"]
                        stg["charge"] += ((1.0 if loud else 0.0) - stg["charge"]) * (
                            stg["attack"] if loud else stg["decay"])
                        if stg["charge"] >= stg["engage_at"]:
                            stg["engaged"] = True
                        elif stg["charge"] <= stg["disengage_at"]:
                            stg["engaged"] = False
                        if not (enabled and stg["engaged"]):
                            # cut immediately on disengage: a triac-dimmed inductive
                            # fan hums audibly if left to ramp down through low
                            # voltages, so snap straight to 0 instead of ramping.
                            stg["cur"] = 0.0
                        else:
                            if proportional:
                                # map wind intensity to fan speed, floored at
                                # min_level so the motor keeps spinning.
                                frac = max(0.0, min(1.0, level / max(DMX_FULL_AT, 1e-6)))
                                target = min_level + (on_level - min_level) * frac
                            else:
                                target = on_level
                            stg["cur"] += (target - stg["cur"]) * ramp_alpha
                        frame[stg["ch"] - 1] = int(max(0, min(255, round(stg["cur"]))))
                    if lt:
                        # Breath -> brightness fraction. Resting on the floor when
                        # played softly; rising toward peak as the wind grows. When
                        # disabled (the 'd' toggle), target 0 so the light fades out.
                        norm = max(0.0, min(1.0, level / lt["full_at"]))
                        target = lt["floor"] + (lt["peak"] - lt["floor"]) * norm if enabled else 0.0
                        lt["bright"] += (target - lt["bright"]) * (
                            lt["rise"] if target > lt["bright"] else lt["fall"])
                        for i, ch in enumerate(lt["chans"]):
                            frame[ch - 1] = int(max(0, min(255, round(lt["color"][i] * lt["bright"]))))
                        frame[lt["master"] - 1] = 255
                    if debug:
                        print(f"[dmx] lvl={level:.3f}  " + "  ".join(
                            f"{stg['label']} ch{stg['ch']}={frame[stg['ch'] - 1]:>3} "
                            f"c={stg['charge']:.2f}{'*' if stg['engaged'] else ' '}"
                            for stg in st))
                    s.write(_packet(frame))
                    time.sleep(period)
                # blackout on the way out: the Enttec keeps re-emitting the LAST
                # frame it received on the DMX wire, so if we just stop, a fan that
                # was on stays on forever. Zero EVERY channel and resend a few times
                # so a single dropped frame can't leave a fan spinning.
                blackout = bytearray(512)
                for _ in range(3):
                    try:
                        s.write(_packet(blackout))
                    except serial.SerialException:
                        break
                    time.sleep(period)
        except serial.SerialException as e:
            print(f"[dmx] serial error: {e}; retry in 2s", file=sys.stderr)
            stop.wait(2.0)
