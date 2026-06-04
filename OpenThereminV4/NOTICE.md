# OpenThereminV4 firmware — vendored, modified fork

This directory is a vendored copy of the **OpenThereminV4** firmware by GaudiLabs,
kept here so the theremin-of-winds value-stream feature is reproducible.

- Upstream: https://github.com/GaudiLabs/OpenThereminV4.git
- Forked from commit `71c73ad93ec0a0366fb0e692190a0a3ddb8e5cb7` (2025-12-08)
- License: **GNU GPL v3** (see `LICENSE`). This firmware subtree stays GPLv3; it is
  NOT relicensed under the parent project's MIT license.

## Our modifications

Only `Software/Open_Theremin_V4/` is vendored (the hardware design files under
`Electronics/` are not). Two files were changed (~42 lines) to add a serial
value-stream mode instead of MIDI:

- `build.h` — new `SERIAL_PORT_MODE_STREAM` (115200 baud), selected by default here.
- `application.cpp` — emits `P<pitch> V<vol> M<mute>\n` (~200 Hz) of the firmware's
  already-linearized pitch/volume; throttled off the free-running `timer` WITHOUT
  `resetTimer()` so the button long-press recalibration keeps working.

The parent repo reads this stream by default (see `stream.py`, `--stream` path).

## Build / flash

```
.toolchain/bin/arduino-cli compile --fqbn arduino:avr:uno Software/Open_Theremin_V4
.toolchain/bin/arduino-cli upload -p /dev/ttyUSB0 --fqbn arduino:avr:uno Software/Open_Theremin_V4
```
