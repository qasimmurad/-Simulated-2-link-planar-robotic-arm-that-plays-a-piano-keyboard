"""
keyboard_layout.py

Defines the (x, y) positions of piano keys across a 5-octave span
(C2 to C7, 36 white keys) in the arm's workspace.

The keyboard is modelled with C2 at x=0 and each successive white key
shifted by WHITE_KEY_SPACING to the right.  All units are metres.

White key spacing:  0.023 m (standard piano key width)
White key y:        0.30 m from the arm base
Black key y:        WHITE_KEY_Y - 0.012 m (slightly closer to arm)
Black key x:        centred between its two flanking white keys

Arm base sits directly below the midpoint of the C4–C5 octave so that
the melody-range keys (C4–C5) are centred in the reachable workspace.
With L1=0.20 m and L2=0.15 m, keys within ~0.18 m of BASE are reachable;
extreme octave keys (C2–E3 and A5–C7) fall outside the workspace,
making the reachability figure informative.
"""

WHITE_KEY_SPACING = 0.023   # metres
WHITE_KEY_Y       = 0.30    # metres from arm base
BLACK_KEY_Y       = WHITE_KEY_Y - 0.012
BLACK_KEY_X_OFFSET = 0.5 * WHITE_KEY_SPACING  # centred between white keys

# ---------------------------------------------------------------------------
# Generate keys for octaves 2–6 using global indexing.
# For octave o, the i-th white note (0=C … 6=B) sits at:
#   x = ((o - 2) * 7 + i) * WHITE_KEY_SPACING
# Black keys follow standard piano layout: C#, D#, (no E#), F#, G#, A#, (no B#)
# ---------------------------------------------------------------------------

_OCTAVE_NOTES  = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
_BLACK_OFFSETS = [0, 1, 3, 4, 5]          # left-white-key index within octave
_BLACK_ACCS    = ['C#', 'D#', 'F#', 'G#', 'A#']

WHITE_KEYS: dict = {}
BLACK_KEYS: dict = {}

for _oct in range(2, 7):          # octaves 2, 3, 4, 5, 6
    _base_idx = (_oct - 2) * 7
    for _i, _n in enumerate(_OCTAVE_NOTES):
        WHITE_KEYS[f"{_n}{_oct}"] = ((_base_idx + _i) * WHITE_KEY_SPACING, WHITE_KEY_Y)
    for _off, _acc in zip(_BLACK_OFFSETS, _BLACK_ACCS):
        BLACK_KEYS[f"{_acc}{_oct}"] = ((_base_idx + _off + 0.5) * WHITE_KEY_SPACING, BLACK_KEY_Y)

# Closing C7 (global index 35, completes the 5-octave span of ~0.805 m)
WHITE_KEYS['C7'] = (35 * WHITE_KEY_SPACING, WHITE_KEY_Y)

# Combined lookup
ALL_KEYS = {**WHITE_KEYS, **BLACK_KEYS}

# ---------------------------------------------------------------------------
# Arm parameters — do not modify without re-running all tests
# ---------------------------------------------------------------------------
L1   = 0.20   # link 1 length (metres)
L2   = 0.15   # link 2 length (metres)

# Centre of the C4–C5 span: C4 is at global index 14, C5 at 21 → mid = 17.5
BASE = (17.5 * WHITE_KEY_SPACING, 0.0)
