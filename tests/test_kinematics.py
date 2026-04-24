"""
test_kinematics.py

Unit tests for the forward and inverse kinematics module.
Each test has a known analytic answer so we can verify correctness
independently of any higher-level code.
"""

import numpy as np
import pytest
from src.robotics.kinematics import (
    forward_kinematics, inverse_kinematics, is_reachable, choose_solution
)
from data.keyboard_layout import L1, L2, BASE


def test_fk_straight_out():
    """
    With both joints at zero, tip should be directly right of base
    at distance L1 + L2.
    """
    base, elbow, tip = forward_kinematics(0.0, 0.0)
    bx, by = BASE
    assert np.isclose(tip[0], bx + L1 + L2, atol=1e-6)
    assert np.isclose(tip[1], by, atol=1e-6)


def test_ik_then_fk_roundtrip():
    """
    IK followed by FK should recover the original target position.
    This is the fundamental correctness test.
    """
    from data.keyboard_layout import ALL_KEYS
    for note, target in ALL_KEYS.items():
        result = inverse_kinematics(target)
        if result is not None:
            _, _, tip = forward_kinematics(result[0], result[1])
            assert np.isclose(tip[0], target[0], atol=1e-4), f"FK x mismatch for {note}"
            assert np.isclose(tip[1], target[1], atol=1e-4), f"FK y mismatch for {note}"


def test_unreachable_target():
    """
    A target far beyond L1 + L2 should return None.
    """
    far_target = (100.0, 100.0)
    assert inverse_kinematics(far_target) is None


def test_is_reachable_all_white_keys():
    """
    Melody-range keys (C4–C5 octave) must be reachable.
    Far-octave extremes must NOT be reachable — this confirms the 5-octave
    keyboard produces a meaningful workspace boundary for the reachability
    figure (the whole point of the expansion).
    If melody keys are unreachable, adjust BASE or L1/L2 in keyboard_layout.py.
    """
    from data.keyboard_layout import WHITE_KEYS
    melody_notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    for note in melody_notes:
        assert is_reachable(WHITE_KEYS[note]), (
            f"{note} at {WHITE_KEYS[note]} must be reachable — adjust arm parameters"
        )
    # Extreme keys must be outside the workspace (validates the reachability analysis)
    far_notes = [n for n in ("C2", "D2", "C7", "B6") if n in WHITE_KEYS]
    for note in far_notes:
        assert not is_reachable(WHITE_KEYS[note]), (
            f"{note} at {WHITE_KEYS[note]} should be unreachable at the workspace extreme"
        )


def test_elbow_up_and_down_both_valid():
    """
    For a target in mid-workspace, both IK solutions should exist
    and both should pass the FK roundtrip.
    """
    target = (BASE[0], BASE[1] + 0.25)  # directly above base
    up   = inverse_kinematics(target, elbow_up=True)
    down = inverse_kinematics(target, elbow_up=False)
    assert up is not None
    assert down is not None
    for sol in [up, down]:
        _, _, tip = forward_kinematics(sol[0], sol[1])
        assert np.isclose(tip[0], target[0], atol=1e-4)
        assert np.isclose(tip[1], target[1], atol=1e-4)
