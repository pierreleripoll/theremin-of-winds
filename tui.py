"""Curses TUI: live knob editing and feature toggles.

Redraws at 10 Hz. Mutates State under State.lock. Macro knobs at the top
write through to fine knobs below via property setters on State.
"""
from typing import NamedTuple

from state import State


class Knob(NamedTuple):
    label: str
    attr: str
    lo: float
    hi: float
    step: float
    fmt: str


# Macro knobs at the top write through to fine knobs below.
MACROS = [
    Knob("» wind bright", "brightness",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    Knob("» whistle    ", "whistle",      0.0,   1.0,   0.05, "{:>6.2f}   "),
]
KNOB_DEFS = MACROS + [
    Knob("low band  Fc", "low_fc",     20.0,  500.0,  10.0,  "{:>6.0f} Hz"),
    Knob("low band   Q", "low_q",       0.3,    5.0,   0.1,  "{:>6.2f}   "),
    Knob("high band Fc", "high_fc",  1000.0, 8000.0, 100.0,  "{:>6.0f} Hz"),
    Knob("high band  Q", "high_q",      1.0,   20.0,   0.5,  "{:>6.2f}   "),
    Knob("high band  G", "high_band_gain",0.0,  1.0,   0.05, "{:>6.2f}   "),
    Knob("mid Fc lo   ", "mid_fc_lo", 100.0,  800.0,  25.0,  "{:>6.0f} Hz"),
    Knob("mid Fc hi   ", "mid_fc_hi", 800.0, 5000.0, 100.0,  "{:>6.0f} Hz"),
    Knob("mid Q max   ", "mid_q_max",   0.0,   8.0,   0.25, "{:>6.2f}   "),
    Knob("gust depth  ", "gust_depth",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    Knob("gust tau    ", "gust_tau_s",   0.2,   8.0,   0.2,  "{:>6.1f} s "),
    Knob("Q drift     ", "q_drift_depth",0.0,   0.5,   0.02, "{:>6.2f}   "),
    Knob("Q drift tau ", "q_drift_tau_s",0.5,   8.0,   0.5,  "{:>6.1f} s "),
    Knob("drive       ", "drive",        0.5,  10.0,   0.2,  "{:>6.2f}   "),
    Knob("tone level  ", "tone_level",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    Knob("bourdon Q   ", "bourdon_q",    2.0,  30.0,   1.0,  "{:>6.1f}   "),
    Knob("organ octave", "organ_octave",-3.0,   0.0,   1.0,  "{:>+6.0f}   "),
    Knob("organ bright", "organ_brightness", 0.0, 1.0,  0.05, "{:>6.2f}   "),
    Knob("organ air   ", "organ_air",    0.0,   1.0,   0.05, "{:>6.2f}   "),
    Knob("organ wind  ", "organ_wind",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    Knob("organ level ", "organ_level",  0.0,   1.5,   0.05, "{:>6.2f}   "),
    Knob("organ thresh", "organ_threshold", 0.0, 1.0,  0.05, "{:>6.2f}   "),
    Knob("organ rise  ", "organ_rise_s", 0.5,  20.0,   0.5,  "{:>6.1f} s "),
    Knob("organ fall  ", "organ_fall_s", 0.5,  20.0,   0.5,  "{:>6.1f} s "),
    Knob("trem depth  ", "trem_depth",   0.0,   0.5,   0.02, "{:>6.2f}   "),
    Knob("trem rate   ", "trem_rate",    3.0,   8.0,   0.1,  "{:>6.2f} Hz"),
    Knob("trem pitch  ", "trem_pitch",   0.0,  0.02,   0.001,"{:>6.3f}   "),
    Knob("pan floor   ", "pan_floor",    0.0,   0.5,   0.02, "{:>6.2f}   "),
    Knob("vol curve   ", "vol_curve",    1.0,   6.0,   0.25, "{:>6.2f}   "),
    Knob("attack      ", "attack_s",     0.0,  10.0,   0.2,  "{:>6.1f} s "),
    Knob("release     ", "release_s",    0.2,  10.0,   0.2,  "{:>6.1f} s "),
]
MACRO_ATTRS = {k.attr for k in MACROS}

THIRD_LABELS = {0: "off", 1: "min", 2: "maj"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _scroll(sel: int, top: int, avail: int, n: int) -> tuple[int, int]:
    """Scroll the knob list so the selected knob stays on screen. Returns the
    (top index, number visible) for a viewport of `avail` rows over `n` knobs."""
    if sel < top:
        top = sel
    elif sel >= top + avail:
        top = sel - avail + 1
    top = max(0, min(top, max(0, n - avail)))
    return top, min(avail, n - top)


def _put(stdscr, y: int, x: int, text: str, width: int, attr: int = 0):
    """addstr that clips to the window and never raises. The knob list can be
    taller than the terminal; rather than let one off-screen write abort the whole
    frame (which desyncs the display from the selection), we skip what won't fit."""
    if y < 0 or x < 0 or x >= width:
        return
    try:
        stdscr.addnstr(y, x, text, max(0, width - x - 1), attr)
    except Exception:
        pass


def tui_loop(stdscr, state: State, port: str, baud: int, fake: bool = False):
    import curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 10 Hz redraw

    KNOBS_TOP = 6
    FOOTER_H = 6  # blank + live + note + amp + blank + help
    sel = 0
    top = 0  # first visible knob; scrolls to keep `sel` on screen on short terminals
    while True:
        stdscr.erase()
        with state.lock:
            u3, ug = state.use_3band, state.use_gust
            u5, tm = state.use_fifth, state.third_mode
            org = state.organ_mode
            usp = state.spatial_mode
            apl = state.autoplay_on
            dmxa, dmxon = state.dmx_available, state.dmx_on
            cp = state.cur_position
            knob_vals = [getattr(state, k.attr) for k in KNOB_DEFS]
            note = state.note
            bend = state.pitch_bend
            cf = state.cur_freq
            ca = state.cur_amp
            cc = state.last_cc
            tilt = state.cur_tilt

        max_y, max_x = stdscr.getmaxyx()
        n = len(KNOB_DEFS)
        avail = max(1, max_y - KNOBS_TOP - FOOTER_H)  # knob rows that fit
        top, visible = _scroll(sel, top, avail, n)

        _put(stdscr, 0, 0, "─── theremin wind ──────────────────────────────", max_x)
        _put(stdscr, 1, 0,
             f"input: trackpad fake — {port}" if fake else f"midi: {port} @ {baud}", max_x)
        feat = (f"features:  [{'x' if u3 else ' '}] 3-band (3)   "
                f"[{'x' if ug else ' '}] gust (g)   "
                f"[{'x' if u5 else ' '}] fifth (5)   "
                f"third: {THIRD_LABELS[tm]} (t)   "
                f"[{'x' if org else ' '}] organ (o)   "
                f"[{'x' if usp else ' '}] spatial (s)   "
                f"[{'x' if apl else ' '}] autoplay (a)")
        if dmxa:
            feat += f"   [{'x' if dmxon else ' '}] dmx fan (d)"
        _put(stdscr, 3, 0, feat, max_x)

        hdr = "knobs:  (» = macro)"
        if top > 0:
            hdr += "   ↑ more"
        if top + visible < n:
            hdr += "   ↓ more"
        _put(stdscr, 5, 0, hdr, max_x)
        for vi in range(visible):
            i = top + vi
            k = KNOB_DEFS[i]
            marker = "▶" if i == sel else " "
            line = f" {marker} {k.label}  {k.fmt.format(knob_vals[i])}"
            attrs = curses.A_BOLD if k.attr in MACRO_ATTRS else 0
            if i == sel:
                attrs |= curses.A_REVERSE
            _put(stdscr, KNOBS_TOP + vi, 0, line, max_x, attrs)

        frow = KNOBS_TOP + visible + 1  # footer sits just below the visible knobs
        _put(stdscr, frow, 0, "live:", max_x)
        _put(stdscr, frow + 1, 2,
             f"note={str(note) if note is not None else '--':>3}  "
             f"bend={bend:>+6}  freq={cf:>6.1f} Hz", max_x)
        spatial_str = f"  pos={cp:.2f}" if usp else ""
        cc_str = f"last_cc={cc[0]}={cc[1]}" if cc else "last_cc=--"
        _put(stdscr, frow + 2, 2,
             f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  {cc_str}", max_x)
        toggles = "3/g/5/t/o/s/a/d" if dmxa else "3/g/5/t/o/s/a"
        _put(stdscr, frow + 4, 0,
             f"↑↓ select   ←→ adjust   {toggles} toggle   q quit", max_x)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break
        if key == -1:
            continue
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            break

        with state.lock:
            if key == ord("3"):
                state.use_3band = not state.use_3band
            elif key == ord("g"):
                state.use_gust = not state.use_gust
            elif key == ord("5"):
                state.use_fifth = not state.use_fifth
            elif key == ord("t"):
                state.third_mode = (state.third_mode + 1) % 3
            elif key == ord("o"):
                state.organ_mode = not state.organ_mode
            elif key == ord("s"):
                state.spatial_mode = not state.spatial_mode
                # re-derive freq/position from current MIDI state under the new mode
                state.recompute_freq()
            elif key == ord("a"):
                state.autoplay_on = not state.autoplay_on
            elif key == ord("d") and state.dmx_available:
                state.dmx_on = not state.dmx_on
            elif key in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_LEFT, ord("h"), curses.KEY_RIGHT, ord("l")):
                k = KNOB_DEFS[sel]
                delta = -k.step if key in (curses.KEY_LEFT, ord("h")) else k.step
                # macro attrs (brightness, whistle) are properties that fan out
                # to fine knobs via their setter — no explicit apply_X needed.
                setattr(state, k.attr, _clamp(getattr(state, k.attr) + delta, k.lo, k.hi))
