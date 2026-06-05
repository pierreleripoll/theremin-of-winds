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
    desc: str  # one-line help shown when this knob is selected


# Macro knobs at the top write through to fine knobs below.
MACROS = [
    Knob("» wind bright", "brightness",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Macro brillance : 0 = sombre/calme, 1 = vif/orageux (pilote gain aigu + drive)"),
    Knob("» whistle    ", "whistle",      0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Macro sifflement : 0 = sans resonance, 1 = plein (pilote Q aigu + Q medium)"),
    Knob("» storm      ", "storm",        0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Macro tempete : 0 = le volume ne change que le niveau, 1 = la brise devient tempete (volume fort = vent plus vif/rafaleux/sature)"),
]
KNOB_DEFS = MACROS + [
    Knob("low band  Fc", "low_fc",     20.0,  500.0,  10.0,  "{:>6.0f} Hz",
         "Frequence centrale de la bande grave (le corps du vent)"),
    Knob("low band   Q", "low_q",       0.3,    5.0,   0.1,  "{:>6.2f}   ",
         "Resonance de la bande grave (haut = plus etroit et pointu)"),
    Knob("high band Fc", "high_fc",  1000.0, 8000.0, 100.0,  "{:>6.0f} Hz",
         "Frequence centrale de la bande aigue (le sifflement du haut)"),
    Knob("high band  Q", "high_q",      1.0,   20.0,   0.5,  "{:>6.2f}   ",
         "Resonance de la bande aigue (haut = plus siffleur)"),
    Knob("high band  G", "high_band_gain",0.0,  1.0,   0.05, "{:>6.2f}   ",
         "Niveau de la bande aigue (0 = pas de 'sizzle' dans le haut)"),
    Knob("mid Fc lo   ", "mid_fc_lo", 100.0,  800.0,  25.0,  "{:>6.0f} Hz",
         "Frequence du medium quand la main est en bas (vent grave)"),
    Knob("mid Fc hi   ", "mid_fc_hi", 800.0, 5000.0, 100.0,  "{:>6.0f} Hz",
         "Frequence du medium quand la main est en haut (vent aigu)"),
    Knob("mid Q max   ", "mid_q_max",   0.0,   8.0,   0.25, "{:>6.2f}   ",
         "Resonance max du medium selon le volume (haut = plus siffleur quand ca joue fort)"),
    Knob("gust depth  ", "gust_depth",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Profondeur des rafales (modulation lente et aleatoire du volume)"),
    Knob("gust tau    ", "gust_tau_s",   0.2,   8.0,   0.2,  "{:>6.1f} s ",
         "Duree d'une rafale (constante de temps : plus haut = rafales plus lentes)"),
    Knob("Q drift     ", "q_drift_depth",0.0,   0.5,   0.02, "{:>6.2f}   ",
         "Profondeur de la derive de resonance (donne de la vie au timbre)"),
    Knob("Q drift tau ", "q_drift_tau_s",0.5,   8.0,   0.5,  "{:>6.1f} s ",
         "Vitesse de la derive de resonance (plus haut = derive plus lente)"),
    Knob("drive       ", "drive",        0.5,  10.0,   0.2,  "{:>6.2f}   ",
         "Saturation tanh : 1.2 = transparent, >3 = chaud, >8 = sature ('horreur')"),
    Knob("tone level  ", "tone_level",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Niveau du bourdon : voix sifflee accordee ajoutee au vent (0 = absente)"),
    Knob("bourdon Q   ", "bourdon_q",    2.0,  30.0,   1.0,  "{:>6.1f}   ",
         "Finesse du bourdon (haut = sifflement etroit/accorde, bas = plus aerien)"),
    Knob("organ octave", "organ_octave",-3.0,   0.0,   1.0,  "{:>+6.0f}   ",
         "Octave de la note grave de l'orgue (-3 = pedale tres grave)"),
    Knob("organ bright", "organ_brightness", 0.0, 1.0,  0.05, "{:>6.2f}   ",
         "Brillance de l'orgue : 0 = sombre/grave, 1 = plein choeur brillant"),
    Knob("organ air   ", "organ_air",    0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Souffle resonant dans le tuyau metallique (l'air dans la pipe)"),
    Knob("organ wind  ", "organ_wind",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Couplage vent vers orgue : a quel point les rafales animent l'orgue"),
    Knob("organ level ", "organ_level",  0.0,   1.5,   0.05, "{:>6.2f}   ",
         "Volume global de l'orgue par rapport au vent (bas = discret)"),
    Knob("organ thresh", "organ_threshold", 0.0, 1.0,  0.05, "{:>6.2f}   ",
         "Seuil de vent au-dessus duquel l'orgue se reveille"),
    Knob("organ rise  ", "organ_rise_s", 0.5,  20.0,   0.5,  "{:>6.1f} s ",
         "Temps de vent fort soutenu avant que l'orgue soit pleinement la"),
    Knob("organ fall  ", "organ_fall_s", 0.5,  20.0,   0.5,  "{:>6.1f} s ",
         "Temps que l'orgue met a s'eteindre quand le vent retombe"),
    Knob("trem depth  ", "trem_depth",   0.0,   0.5,   0.02, "{:>6.2f}   ",
         "Profondeur du tremolo (tremulant d'orgue sur le volume)"),
    Knob("trem rate   ", "trem_rate",    3.0,   8.0,   0.1,  "{:>6.2f} Hz",
         "Vitesse du tremolo (Hz ; un tremulant classique est a 5-6 Hz)"),
    Knob("trem pitch  ", "trem_pitch",   0.0,  0.02,   0.001,"{:>6.3f}   ",
         "Vibrato de hauteur periodique de l'orgue (~5 cents)"),
    Knob("stereo width", "stereo_width",  0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Largeur stereo : 0 = mono (centre), 1 = vent large et enveloppant"),
    Knob("pan floor   ", "pan_floor",    0.0,   0.5,   0.02, "{:>6.2f}   ",
         "Plancher du pan stereo : 0 = pan dur, 0.5 = a peine panoramique"),
    Knob("reverb mix  ", "reverb_mix",    0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Dose de reverb : 0 = sec (off), 1 = tout mouille (queue ajoutee sans latence)"),
    Knob("reverb size ", "reverb_room",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Taille/duree de la reverb : 0 = petite piece, 1 = longue cathedrale"),
    Knob("reverb damp ", "reverb_damping",0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Amortissement des aigus dans la queue : 0 = brillant, 1 = sombre/feutre"),
    Knob("mic gain    ", "mic_gain",      0.0,   4.0,   0.1,  "{:>6.2f}   ",
         "Niveau sec du micro dans la sortie : 0 = aucun (monitoring direct SSL), >1 = amplifie"),
    Knob("mic reverb  ", "mic_send",      0.0,   2.0,   0.05, "{:>6.2f}   ",
         "Niveau de reverb de la voix (independant du 'reverb mix' du vent) : 0 = voix seche"),
    Knob("mic room    ", "mic_room",      0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Taille de la reverb de la voix (independante de celle du vent) : 0 = petite, 1 = cathedrale"),
    Knob("mic damp    ", "mic_damping",   0.0,   1.0,   0.05, "{:>6.2f}   ",
         "Amortissement de la reverb voix : haut = queue sombre (calme aussi le larsen aigu)"),
    Knob("mic gate    ", "mic_gate",      0.0,   0.1,   0.005,"{:>6.3f}   ",
         "Noise gate anti-larsen : coupe le micro sous ce niveau (0 = off ; monter juste au-dessus du bruit)"),
    Knob("mic EQ1 Fc  ", "mic_eq1_fc",   30.0,  500.0,  10.0,  "{:>6.0f} Hz",
         "EQ micro bande 1 (grave) : frequence centrale de la cloche"),
    Knob("mic EQ1 Q   ", "mic_eq1_q",     0.3,   10.0,   0.1,  "{:>6.2f}   ",
         "EQ micro bande 1 : largeur (haut = etroit, pour creuser une resonance/larsen)"),
    Knob("mic EQ1 G   ", "mic_eq1_gain",-18.0,   18.0,   0.5,  "{:>+6.1f} dB",
         "EQ micro bande 1 : gain en dB (0 = plat ; - = creux, + = bosse)"),
    Knob("mic EQ2 Fc  ", "mic_eq2_fc",  100.0, 2000.0,  25.0,  "{:>6.0f} Hz",
         "EQ micro bande 2 (bas-medium) : frequence centrale (boite/boue vers 300-500 Hz)"),
    Knob("mic EQ2 Q   ", "mic_eq2_q",     0.3,   10.0,   0.1,  "{:>6.2f}   ",
         "EQ micro bande 2 : largeur (haut = etroit)"),
    Knob("mic EQ2 G   ", "mic_eq2_gain",-18.0,   18.0,   0.5,  "{:>+6.1f} dB",
         "EQ micro bande 2 : gain en dB (0 = plat)"),
    Knob("mic EQ3 Fc  ", "mic_eq3_fc",  500.0, 8000.0,  50.0,  "{:>6.0f} Hz",
         "EQ micro bande 3 (presence) : frequence centrale (intelligibilite vers 2-4 kHz)"),
    Knob("mic EQ3 Q   ", "mic_eq3_q",     0.3,   10.0,   0.1,  "{:>6.2f}   ",
         "EQ micro bande 3 : largeur (haut = etroit)"),
    Knob("mic EQ3 G   ", "mic_eq3_gain",-18.0,   18.0,   0.5,  "{:>+6.1f} dB",
         "EQ micro bande 3 : gain en dB (0 = plat)"),
    Knob("mic EQ4 Fc  ", "mic_eq4_fc", 2000.0,16000.0, 100.0,  "{:>6.0f} Hz",
         "EQ micro bande 4 (air) : frequence centrale (brillance/sifflantes vers 6-10 kHz)"),
    Knob("mic EQ4 Q   ", "mic_eq4_q",     0.3,   10.0,   0.1,  "{:>6.2f}   ",
         "EQ micro bande 4 : largeur (haut = etroit, pour pincer une sifflante)"),
    Knob("mic EQ4 G   ", "mic_eq4_gain",-18.0,   18.0,   0.5,  "{:>+6.1f} dB",
         "EQ micro bande 4 : gain en dB (0 = plat)"),
    Knob("vol curve   ", "vol_curve",    1.0,   6.0,   0.25, "{:>6.2f}   ",
         "Courbe de volume : >1 etire la zone douce pour un controle fin des faibles"),
    Knob("attack      ", "attack_s",     0.0,  10.0,   0.2,  "{:>6.1f} s ",
         "Temps de montee en regime du vent quand une main revient"),
    Knob("release     ", "release_s",    0.2,  10.0,   0.2,  "{:>6.1f} s ",
         "Temps de fondu du vent quand la main s'en va"),
    Knob("amp rise    ", "amp_rise_s",   0.0,   3.0,   0.1,  "{:>6.1f} s ",
         "Inertie de montee du volume (slew anti-rafale) : 0 = vent instantane, haut = montee molle/securisee"),
    Knob("inertia add ", "inertia_add_s",0.0,   3.0,   0.1,  "{:>6.1f} s ",
         "Inertie de redemarrage 'machine a vent' apres un arret : 0 = reprise vive, haut = demarrage lent"),
]
MACRO_ATTRS = {k.attr for k in MACROS}

THIRD_LABELS = {0: "off", 1: "min", 2: "maj"}

# Solo: isolate one audio layer to learn what each knob does. (state key, label, key).
# Several can be on at once; none on = everything plays. Band solos (grave/medium/aigu)
# only separate in 3-band mode; in single-band mode the whole wind sits under "grave".
SOLO_DEFS = [
    ("low",     "grave",   "z"),
    ("mid",     "medium",  "x"),
    ("high",    "aigu",    "v"),
    ("bourdon", "bourdon", "b"),
    ("organ",   "orgue",   "n"),
]
SOLO_KEYS = {ord(key): name for name, _lbl, key in SOLO_DEFS}


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

    import presets
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)  # 10 Hz redraw

    KNOBS_TOP = 6
    FOOTER_H = 8  # desc + blank + live + note + amp + blank + help (+1 gap above)
    sel = 0
    top = 0  # first visible knob; scrolls to keep `sel` on screen on short terminals
    status = ""        # transient footer message (e.g. "saved preset")
    status_frames = 0  # redraws left to keep showing `status`
    while True:
        stdscr.erase()
        with state.lock:
            u3, ug = state.use_3band, state.use_gust
            u5, tm = state.use_fifth, state.third_mode
            org = state.organ_mode
            usp = state.spatial_mode
            apl = state.autoplay_on
            aplm = state.autoplay_max
            mut = state.muted
            solo = set(state.solo)
            dmxa, dmxon = state.dmx_available, state.dmx_on
            cp = state.cur_position
            knob_vals = [getattr(state, k.attr) for k in KNOB_DEFS]
            note = state.note
            bend = state.pitch_bend
            cf = state.cur_freq
            ca = state.cur_amp
            cc = state.last_cc
            tilt = state.cur_tilt
            rest_amp, rest_note = state.rest_amp, state.rest_note
            calibrating = state.calibrating

        max_y, max_x = stdscr.getmaxyx()
        n = len(KNOB_DEFS)
        # two-column grid, filled column-major: left column is the first n_rows
        # knobs, right column the rest. ↑↓ walks the linear index, so it runs down
        # the left column then continues at the top of the right one.
        n_rows = (n + 1) // 2
        sel_row = sel if sel < n_rows else sel - n_rows
        avail = max(1, max_y - KNOBS_TOP - FOOTER_H)  # grid rows that fit
        top, visible = _scroll(sel_row, top, avail, n_rows)
        col_w = max(24, max_x // 2)

        _put(stdscr, 0, 0, "─── theremin wind ──────────────────────────────", max_x)
        _put(stdscr, 1, 0,
             f"input: trackpad fake — {port}" if fake else f"midi: {port} @ {baud}", max_x)
        # Two compact lines so every toggle stays visible on a narrow (80-col)
        # terminal -- the old single line ran ~177 chars and clipped the last
        # toggles (incl. dmx fan) off the right edge.
        def tog(on: bool, label: str, key: str) -> str:
            return f"[{'x' if on else ' '}]{label}({key})"
        line1 = "  ".join((
            tog(u3, "3band", "3"), tog(ug, "gust", "g"), tog(u5, "fifth", "5"),
            f"third:{THIRD_LABELS[tm]}(t)", tog(org, "organ", "o"), tog(usp, "spatial", "s"),
        ))
        line2 = [tog(apl, "autoplay", "a"), tog(aplm, "max-vol", "A"), tog(mut, "mute", "m")]
        if dmxa:
            line2.append(tog(dmxon, "dmx fan", "d"))
        solo_line = "solo: " + "  ".join(
            tog(name in solo, lbl, key) for name, lbl, key in SOLO_DEFS)
        if not solo:
            solo_line += "   (rien = tout)"
        _put(stdscr, 2, 0, line1, max_x)
        _put(stdscr, 3, 0, "  ".join(line2), max_x)
        _put(stdscr, 4, 0, solo_line, max_x)

        hdr = "knobs:  (» = macro)"
        if top > 0:
            hdr += "   ↑ more"
        if top + visible < n_rows:
            hdr += "   ↓ more"
        _put(stdscr, 5, 0, hdr, max_x)

        def draw_cell(i: int, x: int, cell_w: int):
            k = KNOB_DEFS[i]
            marker = "▶" if i == sel else " "
            line = f" {marker} {k.label}  {k.fmt.format(knob_vals[i])}"
            attrs = curses.A_BOLD if k.attr in MACRO_ATTRS else 0
            if i == sel:
                attrs |= curses.A_REVERSE
            _put(stdscr, KNOBS_TOP + vi, x, line, cell_w, attrs)

        for vi in range(visible):
            r = top + vi
            draw_cell(r, 0, col_w)            # left column
            if r + n_rows < n:
                draw_cell(r + n_rows, col_w, max_x)  # right column

        frow = KNOBS_TOP + visible + 1  # footer sits just below the visible knobs
        ksel = KNOB_DEFS[sel]
        _put(stdscr, frow, 0, f" {ksel.label.strip()}: {ksel.desc}",
             max_x, curses.A_BOLD)
        _put(stdscr, frow + 2, 0, "live:", max_x)
        _put(stdscr, frow + 3, 2,
             f"note={str(note) if note is not None else '--':>3}  "
             f"bend={bend:>+6}  freq={cf:>6.1f} Hz", max_x)
        spatial_str = f"  pos={cp:.2f}" if usp else ""
        cc_str = f"last_cc={cc[0]}={cc[1]}" if cc else "last_cc=--"
        if calibrating:
            rest_str = "rest=CALIBRATING"
        elif rest_amp is None:
            rest_str = "rest=--"
        else:
            rn = f"{rest_note:.0f}" if rest_note is not None else "--"
            rest_str = f"rest={rest_amp:.2f}/{rn}"
        _put(stdscr, frow + 4, 2,
             f"amp={ca:.2f}  tilt={tilt:.2f}{spatial_str}  {cc_str}  {rest_str}", max_x)
        if status_frames > 0:
            _put(stdscr, frow + 5, 0, status, max_x, curses.A_BOLD)
            status_frames -= 1
        toggles = "3/g/5/t/o/s/a/d" if dmxa else "3/g/5/t/o/s/a"
        _put(stdscr, frow + 6, 0,
             f"↑↓ sel  ←→ adj  {toggles} tog  z/x/v/b/n solo  c calib  w save  q quit",
             max_x)
        stdscr.refresh()

        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            break
        if key == -1:
            continue
        if key in (ord("q"), ord("Q"), 27):  # q or ESC
            break
        if key in (ord("w"), ord("W")):  # save knobs over the preset file
            # save_preset takes state.lock itself, so do it outside the lock block.
            presets.save_preset(state)
            status, status_frames = f"saved {presets.PRESET_PATH.name}", 20
            continue
        if key in (ord("c"), ord("C")):  # recalibrate the rest point (stand back!)
            with state.lock:
                state.calibrate_request = True
            status, status_frames = "calibrating rest... (stand back)", 20
            continue

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
                if not state.autoplay_on:
                    state.autoplay_max = False
            elif key == ord("A"):
                state.autoplay_max = not state.autoplay_max
                if state.autoplay_max:
                    state.autoplay_on = True
            elif key == ord("m"):
                state.muted = not state.muted
            elif key in SOLO_KEYS:
                name = SOLO_KEYS[key]
                state.solo.discard(name) if name in state.solo else state.solo.add(name)
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
