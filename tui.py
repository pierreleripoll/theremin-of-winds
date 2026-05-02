"""Curses TUI: live knob editing and feature toggles.

Redraws at 10 Hz. Mutates State under State.lock. Macro knobs at the top
write through to fine knobs below (see State.apply_brightness / apply_whistle).
"""
from state import State

# (label, attr, min, max, step, fmt). Macro knobs at the top write through to fine
# knobs below.
MACROS = [
    ("» brightness ", "brightness",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("» whistle    ", "whistle",      0.0,   1.0,   0.05, "{:>6.2f}   "),
]
KNOB_DEFS = MACROS + [
    ("low band  Fc", "low_fc",     20.0,  500.0,  10.0,  "{:>6.0f} Hz"),
    ("low band   Q", "low_q",       0.3,    5.0,   0.1,  "{:>6.2f}   "),
    ("high band Fc", "high_fc",  1000.0, 8000.0, 100.0,  "{:>6.0f} Hz"),
    ("high band  Q", "high_q",      1.0,   20.0,   0.5,  "{:>6.2f}   "),
    ("high band  G", "high_band_gain",0.0,  1.0,   0.05, "{:>6.2f}   "),
    ("mid Fc lo   ", "mid_fc_lo", 100.0,  800.0,  25.0,  "{:>6.0f} Hz"),
    ("mid Fc hi   ", "mid_fc_hi", 800.0, 5000.0, 100.0,  "{:>6.0f} Hz"),
    ("mid Q max   ", "mid_q_max",   0.0,   8.0,   0.25, "{:>6.2f}   "),
    ("gust depth  ", "gust_depth",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("gust tau    ", "gust_tau_s",   0.2,   8.0,   0.2,  "{:>6.1f} s "),
    ("Q drift     ", "q_drift_depth",0.0,   0.5,   0.02, "{:>6.2f}   "),
    ("Q drift tau ", "q_drift_tau_s",0.5,   8.0,   0.5,  "{:>6.1f} s "),
    ("drive       ", "drive",        0.5,  10.0,   0.2,  "{:>6.2f}   "),
    ("tone level  ", "tone_level",   0.0,   1.0,   0.05, "{:>6.2f}   "),
    ("bourdon Q   ", "bourdon_q",    2.0,  30.0,   1.0,  "{:>6.1f}   "),
    ("pan floor   ", "pan_floor",    0.0,   0.5,   0.02, "{:>6.2f}   "),
]
MACRO_ATTRS = {a for _, a, *_ in MACROS}

THIRD_LABELS = {0: "off", 1: "min", 2: "maj"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def tui_loop(stdscr, state: State, port: str, baud: int, fake: bool = False):
    import curses
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 10 Hz redraw

    sel = 0
    while True:
        stdscr.erase()
        with state.lock:
            u3, ug = state.use_3band, state.use_gust
            u5, tm = state.use_fifth, state.third_mode
            usp = state.spatial_mode
            cp = state.cur_position
            knob_vals = [getattr(state, a) for _, a, *_ in KNOB_DEFS]
            note = state.note
            bend = state.pitch_bend
            cf = state.cur_freq
            ca = state.cur_amp
            cc = state.last_cc
            tilt = state.cur_tilt

        try:
            stdscr.addstr(0, 0, "─── theremin wind ──────────────────────────────")
            if fake:
                stdscr.addstr(1, 0, f"input: trackpad fake — {port}")
            else:
                stdscr.addstr(1, 0, f"midi: {port} @ {baud}")

            stdscr.addstr(3, 0, "features:  ")
            stdscr.addstr(f"[{'x' if u3 else ' '}] 3-band (3)   ")
            stdscr.addstr(f"[{'x' if ug else ' '}] gust (g)   ")
            stdscr.addstr(f"[{'x' if u5 else ' '}] fifth (5)   ")
            stdscr.addstr(f"third: {THIRD_LABELS[tm]} (t)   ")
            stdscr.addstr(f"[{'x' if usp else ' '}] spatial (s)")

            stdscr.addstr(5, 0, "knobs:  (» = macro: writes through to fine knobs below)")
            for i, ((lab, attr, _, _, _, fmt), v) in enumerate(zip(KNOB_DEFS, knob_vals)):
                marker = "▶" if i == sel else " "
                line = f" {marker} {lab}  {fmt.format(v)}"
                attrs = curses.A_BOLD if attr in MACRO_ATTRS else 0
                if i == sel:
                    attrs |= curses.A_REVERSE
                stdscr.addstr(6 + i, 0, line, attrs)

            row = 6 + len(KNOB_DEFS) + 1
            stdscr.addstr(row, 0, "live:")
            stdscr.addstr(row + 1, 2,
                          f"note={str(note) if note is not None else '--':>3}  "
                          f"bend={bend:>+6}  freq={cf:>6.1f} Hz")
            spatial_str = f"  pos={cp:.2f}" if usp else ""
            stdscr.addstr(row + 2, 2,
                          f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  "
                          f"last_cc={cc[0]}={cc[1]}" if cc else
                          f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  last_cc=--")

            stdscr.addstr(row + 4, 0,
                          "↑↓ select knob   ←→ adjust   3/g/5/t/s toggle   q quit")
        except curses.error:
            pass  # window too small; skip this frame
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
            elif key == ord("s"):
                state.spatial_mode = not state.spatial_mode
                # re-derive freq/position from current MIDI state under the new mode
                state._recompute_freq()
            elif key in (curses.KEY_UP, ord("k")):
                sel = (sel - 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_DOWN, ord("j")):
                sel = (sel + 1) % len(KNOB_DEFS)
            elif key in (curses.KEY_LEFT, ord("h"), curses.KEY_RIGHT, ord("l")):
                _, attr, lo, hi, step, _ = KNOB_DEFS[sel]
                delta = -step if key in (curses.KEY_LEFT, ord("h")) else step
                setattr(state, attr, _clamp(getattr(state, attr) + delta, lo, hi))
                if attr == "brightness":
                    state.apply_brightness()
                elif attr == "whistle":
                    state.apply_whistle()
