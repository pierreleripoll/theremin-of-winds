#!/usr/bin/env python3
"""Dump raw bytes from a serial port for ~3 seconds, at multiple baud rates,
to figure out what the OpenTheremin is actually sending and at what rate.

Usage:
  python sniff_serial.py                       # try 115200 then 31250
  python sniff_serial.py --baud 115200
  python sniff_serial.py --port /dev/ttyUSB0 --baud 115200 --duration 3
"""
import argparse
import collections
import sys
import time

import serial


def looks_like_midi(buf: bytes) -> tuple[int, str]:
    """Score how MIDI-like a buffer is. Returns (score, summary)."""
    if not buf:
        return 0, "empty"
    status_bytes = sum(1 for b in buf if 0x80 <= b <= 0xEF)
    data_bytes = sum(1 for b in buf if b < 0x80)
    realtime = sum(1 for b in buf if b >= 0xF8)

    # Check for plausible channel-message status nibbles
    nibble_hist = collections.Counter(b & 0xF0 for b in buf if b & 0x80)
    common_kinds = {0x80, 0x90, 0xB0, 0xE0}
    plausible = sum(c for k, c in nibble_hist.items() if k in common_kinds)

    score = plausible * 3 + data_bytes - realtime
    summary = (
        f"len={len(buf)} status={status_bytes} data={data_bytes} "
        f"plausible-status={plausible} top-nibbles={dict(nibble_hist.most_common(5))}"
    )
    return score, summary


def sniff(port: str, baud: int, duration: float) -> bytes:
    print(f"\n--- {port} @ {baud} baud, {duration:.1f}s ---")
    buf = bytearray()
    try:
        with serial.Serial(port, baud, timeout=0.1) as s:
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                buf.extend(s.read(256))
    except serial.SerialException as e:
        print(f"open failed: {e}")
        return b""

    if not buf:
        print("(no bytes received — is the theremin powered? play it during the sniff!)")
        return bytes(buf)

    score, summary = looks_like_midi(bytes(buf))
    print(summary)
    print(f"midi-likelihood score: {score}")
    print("first 64 bytes hex:", " ".join(f"{b:02x}" for b in buf[:64]))
    if len(buf) > 64:
        print("last  64 bytes hex:", " ".join(f"{b:02x}" for b in buf[-64:]))
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, help="single baud rate to test")
    ap.add_argument("--duration", type=float, default=3.0)
    args = ap.parse_args()

    print("Move your hands around the antennas during the sniff!")
    time.sleep(1.0)

    rates = [args.baud] if args.baud else [115200, 31250, 38400, 57600, 9600]
    results = []
    for r in rates:
        buf = sniff(args.port, r, args.duration)
        score, _ = looks_like_midi(buf)
        results.append((r, score, len(buf)))

    print("\n=== summary ===")
    for r, score, n in sorted(results, key=lambda x: -x[1]):
        print(f"  {r:>6} baud: score={score:>4}  ({n} bytes)")


if __name__ == "__main__":
    main()
