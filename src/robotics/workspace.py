"""
workspace.py

Reachability analysis for the 2-link planar arm.
"""

import numpy as np
from src.robotics.kinematics import is_reachable
from data.keyboard_layout import ALL_KEYS, L1, L2, BASE


def reachable_keys():
    """Return {note: pos} for every key the arm can reach."""
    return {note: pos for note, pos in ALL_KEYS.items() if is_reachable(pos)}


def unreachable_keys():
    """Return {note: pos} for every key outside the workspace."""
    return {note: pos for note, pos in ALL_KEYS.items() if not is_reachable(pos)}


def workspace_grid(resolution=100):
    """
    Return a list of (x, y) points that lie inside the reachable annulus.
    Useful for visualising the workspace boundary.
    """
    r_max = L1 + L2
    bx, by = BASE
    xs = np.linspace(bx - r_max, bx + r_max, resolution)
    ys = np.linspace(by - r_max, by + r_max, resolution)
    return [
        (float(x), float(y))
        for x in xs
        for y in ys
        if is_reachable((float(x), float(y)))
    ]
