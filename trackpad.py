"""Trackpad fake-input: drive the synth from a touchpad when no theremin
is plugged in. evdev-based; Linux only."""
import glob

from state import State


def find_touchpad():
    """Return the first evdev device that looks like a touchpad, or None.

    evdev.list_devices() silently drops devices the current user can't read,
    so an empty result usually means a missing 'input' group, not "no devices".
    Detect that and raise PermissionError so the caller can show a useful hint.
    """
    import evdev
    from evdev import ecodes
    paths = evdev.list_devices()
    if not paths and glob.glob("/dev/input/event*"):
        raise PermissionError("/dev/input/event* exists but is unreadable by this user")
    for path in paths:
        try:
            dev = evdev.InputDevice(path)
        except (PermissionError, OSError):
            continue
        caps = dev.capabilities()
        abs_codes = {c[0] for c in caps.get(ecodes.EV_ABS, [])}
        key_codes = set(caps.get(ecodes.EV_KEY, []))
        if (ecodes.ABS_X in abs_codes and ecodes.ABS_Y in abs_codes
                and ecodes.BTN_TOUCH in key_codes
                and ecodes.INPUT_PROP_POINTER in dev.input_props()):
            return dev
        dev.close()
    return None


def open_touchpad(dev_path: str | None):
    """Open the requested touchpad, or autodetect one.

    Raises ImportError if `evdev` isn't installed, PermissionError if /dev/input
    devices exist but aren't readable, FileNotFoundError if no touchpad matches.
    """
    import evdev
    if dev_path:
        return evdev.InputDevice(dev_path)
    tpad = find_touchpad()
    if tpad is None:
        raise FileNotFoundError("no touchpad found")
    return tpad


def trackpad_loop(state: State, dev, grab: bool = False):
    """Read absolute X/Y from a touchpad and drive State.fake_xy.

    Trackpad space → synth: x_min..x_max → freq (log), y_min..y_max → amp
    (inverted so top-of-pad = loud). Lifting the finger drops amp to 0.
    """
    from evdev import ecodes
    ax = dev.absinfo(ecodes.ABS_X)
    ay = dev.absinfo(ecodes.ABS_Y)
    x_min, x_max = ax.min, ax.max
    y_min, y_max = ay.min, ay.max
    x_span = max(1, x_max - x_min)
    y_span = max(1, y_max - y_min)

    cur_x = (x_min + x_max) / 2
    cur_y = (y_min + y_max) / 2
    touching = False

    if grab:
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
                    with state.lock:
                        state.target_amp = 0.0
            elif ev.type == ecodes.EV_SYN and touching:
                x_norm = (cur_x - x_min) / x_span
                y_norm = 1.0 - (cur_y - y_min) / y_span
                with state.lock:
                    state.fake_xy(x_norm, y_norm)
    finally:
        if grab:
            try:
                dev.ungrab()
            except OSError:
                pass
