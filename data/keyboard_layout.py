"""
keyboard_layout.py

Defines the (x, y) positions of piano keys in the arm's workspace.

The keyboard is modelled as a single octave (C4 to C5, 8 white keys)
laid flat along the x-axis. The arm base sits at the origin (0, 0),
above the keyboard. All units are in metres.

White key spacing: 0.023 m (23mm, standard piano key width)
White key y-position: 0.3 m (30cm from arm base)
Black keys are offset 0.012 m forward (toward the arm) and raised 0.01 m.
"""

# White keys: C4, D4, E4, F4, G4, A4, B4, C5
WHITE_KEY_SPACING = 0.023  # metres
WHITE_KEY_Y = 0.30          # metres from arm base

WHITE_KEYS = {
    "C4": (0 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "D4": (1 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "E4": (2 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "F4": (3 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "G4": (4 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "A4": (5 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "B4": (6 * WHITE_KEY_SPACING, WHITE_KEY_Y),
    "C5": (7 * WHITE_KEY_SPACING, WHITE_KEY_Y),
}

# Black keys: C#4, D#4, F#4, G#4, A#4
BLACK_KEY_Y = WHITE_KEY_Y - 0.012  # slightly closer to arm
BLACK_KEY_X_OFFSET = 0.5 * WHITE_KEY_SPACING  # centred between white keys

BLACK_KEYS = {
    "C#4": (0.5 * WHITE_KEY_SPACING, BLACK_KEY_Y),
    "D#4": (1.5 * WHITE_KEY_SPACING, BLACK_KEY_Y),
    "F#4": (3.5 * WHITE_KEY_SPACING, BLACK_KEY_Y),
    "G#4": (4.5 * WHITE_KEY_SPACING, BLACK_KEY_Y),
    "A#4": (5.5 * WHITE_KEY_SPACING, BLACK_KEY_Y),
}

# Combined lookup
ALL_KEYS = {**WHITE_KEYS, **BLACK_KEYS}

# Arm parameters — tune these once IK is validated
L1 = 0.20  # length of link 1 in metres
L2 = 0.15  # length of link 2 in metres
BASE = (3.5 * WHITE_KEY_SPACING, 0.0)  # arm base, centred above keyboard
