"""
resolver.py

Maps note names to keyboard positions and provides built-in melody sequences.
"""

from data.keyboard_layout import ALL_KEYS

TWINKLE = [
    "C4", "C4", "G4", "G4", "A4", "A4", "G4",
    "F4", "F4", "E4", "E4", "D4", "D4", "C4",
]

MARY = [
    "E4", "D4", "C4", "D4", "E4", "E4", "E4",
    "D4", "D4", "D4", "E4", "G4", "G4",
]

ODE_TO_JOY = [
    "E4", "E4", "F4", "G4", "G4", "F4", "E4", "D4",
    "C4", "C4", "D4", "E4", "E4", "D4", "D4",
]

MELODIES = {
    "twinkle": TWINKLE,
    "mary": MARY,
    "ode_to_joy": ODE_TO_JOY,
}


def resolve_note(note: str):
    """Return (x, y) position for a note name, or None if unknown."""
    if note in ALL_KEYS:
        return ALL_KEYS[note]
    # Case-insensitive fallback
    note_lower = note.lower()
    for k, v in ALL_KEYS.items():
        if k.lower() == note_lower:
            return v
    return None


def resolve_melody(notes):
    """
    Convert a list of note names to (note, position) pairs.
    Notes not found in ALL_KEYS are silently skipped.
    """
    result = []
    for note in notes:
        pos = resolve_note(note)
        if pos is not None:
            result.append((note, pos))
    return result
