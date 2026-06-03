"""Save / load knob + toggle settings to a JSON preset file.

The TUI 'w' key writes the current knobs over PRESET_PATH; theremin_wind loads
it at startup if present, so a tuning survives between sessions instead of being
re-dialed every time. One file, overwritten in place — a single working patch,
not a library of named presets.
"""
import json
from pathlib import Path

from state import State
from tui import KNOB_DEFS

PRESET_PATH = Path(__file__).resolve().parent / "preset.json"

# Sound-shaping toggles worth persisting alongside the knobs. Excludes runtime /
# hardware modes (autoplay demo, dmx fan) that belong to a session, not a tuning.
TOGGLE_ATTRS = ("use_3band", "use_gust", "use_fifth", "third_mode",
                "organ_mode", "spatial_mode")


def save_preset(state: State, path: Path = PRESET_PATH) -> None:
    with state.lock:
        data = {k.attr: getattr(state, k.attr) for k in KNOB_DEFS}
        data.update({a: getattr(state, a) for a in TOGGLE_ATTRS})
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_preset(state: State, path: Path = PRESET_PATH) -> bool:
    """Apply a saved preset onto state. Returns False if the file is absent."""
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    with state.lock:
        # Set macros first: each fans out to fine knobs via its setter, then the
        # saved fine-knob values below overwrite that fan-out with the exact
        # tuning (KNOB_DEFS lists the two macros before the fine knobs).
        for k in KNOB_DEFS:
            if k.attr in data:
                setattr(state, k.attr, data[k.attr])
        for a in TOGGLE_ATTRS:
            if a in data:
                setattr(state, a, data[a])
    return True
