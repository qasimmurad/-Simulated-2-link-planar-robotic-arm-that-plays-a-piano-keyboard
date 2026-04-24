"""
test_search.py

Unit tests for motion planning (search and heuristics).
"""

import pytest
from src.music.resolver import resolve_melody, TWINKLE, MARY
from src.planning.search import (
    greedy_plan, astar_plan, astar_plan_wide, ucs_plan, total_joint_travel, _wide_configs,
)
from src.planning.heuristics import joint_travel_cost
from data.keyboard_layout import ALL_KEYS


def test_greedy_plan_length():
    note_positions = resolve_melody(TWINKLE)
    plan = greedy_plan(note_positions)
    assert len(plan) == len(note_positions)


def test_greedy_plan_returns_floats():
    note_positions = resolve_melody(TWINKLE)
    for _, t1, t2 in greedy_plan(note_positions):
        assert isinstance(t1, float)
        assert isinstance(t2, float)


def test_astar_plan_length():
    note_positions = resolve_melody(TWINKLE)
    plan = astar_plan(note_positions)
    assert len(plan) == len(note_positions)


def test_astar_no_worse_than_greedy():
    note_positions = resolve_melody(MARY)
    greedy = greedy_plan(note_positions)
    astar = astar_plan(note_positions)
    assert total_joint_travel(astar) <= total_joint_travel(greedy) + 1e-6


def test_joint_travel_zero_same_config():
    assert joint_travel_cost(0.5, 0.3, 0.5, 0.3) == 0.0


def test_joint_travel_positive():
    assert joint_travel_cost(0.0, 0.0, 1.0, 1.0) == pytest.approx(2.0)


def test_greedy_raises_on_unreachable():
    with pytest.raises(ValueError):
        greedy_plan([("X99", (100.0, 100.0))])


def test_astar_raises_on_unreachable():
    with pytest.raises(ValueError):
        astar_plan([("X99", (100.0, 100.0))])


def test_total_joint_travel_empty():
    assert total_joint_travel([]) == 0.0


def test_total_joint_travel_single():
    assert total_joint_travel([("C4", 0.5, 0.3)]) == 0.0


# --- astar_plan_wide tests ---

def test_wide_configs_count():
    """Every reachable key should yield exactly 6 wide configs (2 IK × 3 deltas)."""
    for note, pos in ALL_KEYS.items():
        cfgs = _wide_configs(pos)
        assert len(cfgs) == 6, f"{note}: expected 6 wide configs, got {len(cfgs)}"


def test_wide_configs_are_floats():
    pos = ALL_KEYS["C4"]
    for t1, t2 in _wide_configs(pos):
        assert isinstance(t1, float)
        assert isinstance(t2, float)


def test_wide_configs_delta_zero_matches_ik():
    """The delta=0 configs should match the two standard IK solutions."""
    from src.robotics.kinematics import inverse_kinematics
    import math
    pos = ALL_KEYS["G4"]
    cfgs = _wide_configs(pos)
    ik_up   = inverse_kinematics(pos, elbow_up=True)
    ik_down = inverse_kinematics(pos, elbow_up=False)
    # Each base IK theta1 should appear in the wide config list
    wide_t1s = [t1 for t1, _ in cfgs]
    assert any(math.isclose(t1, ik_up[0],   abs_tol=1e-6) for t1 in wide_t1s)
    assert any(math.isclose(t1, ik_down[0], abs_tol=1e-6) for t1 in wide_t1s)


def test_astar_wide_plan_length():
    note_positions = resolve_melody(TWINKLE)
    plan = astar_plan_wide(note_positions)
    assert len(plan) == len(note_positions)


def test_astar_wide_returns_floats():
    note_positions = resolve_melody(TWINKLE)
    for _, t1, t2 in astar_plan_wide(note_positions):
        assert isinstance(t1, float)
        assert isinstance(t2, float)


def test_astar_wide_beats_greedy():
    """Wide A* must find a strictly cheaper plan than greedy on Twinkle."""
    note_positions = resolve_melody(TWINKLE)
    greedy_cost = total_joint_travel(greedy_plan(note_positions))
    wide_cost   = total_joint_travel(astar_plan_wide(note_positions))
    assert wide_cost < greedy_cost, (
        f"astar_plan_wide ({wide_cost:.4f}) should beat greedy ({greedy_cost:.4f})"
    )


def test_astar_wide_no_worse_than_astar():
    """Wide A* must be at least as good as narrow A* (superset of states)."""
    note_positions = resolve_melody(MARY)
    narrow_cost = total_joint_travel(astar_plan(note_positions))
    wide_cost   = total_joint_travel(astar_plan_wide(note_positions))
    assert wide_cost <= narrow_cost + 1e-6


def test_astar_wide_raises_on_unreachable():
    with pytest.raises(ValueError):
        astar_plan_wide([("X99", (100.0, 100.0))])


# --- ucs_plan tests ---

def test_ucs_plan_length():
    note_positions = resolve_melody(TWINKLE)
    plan = ucs_plan(note_positions)
    assert len(plan) == len(note_positions)


def test_ucs_plan_returns_floats():
    note_positions = resolve_melody(TWINKLE)
    for _, t1, t2 in ucs_plan(note_positions):
        assert isinstance(t1, float)
        assert isinstance(t2, float)


def test_ucs_matches_astar_wide_cost_twinkle():
    """UCS and A*-wide operate on the same state space so must find equal optimal costs."""
    note_positions = resolve_melody(TWINKLE)
    ucs_cost  = total_joint_travel(ucs_plan(note_positions))
    wide_cost = total_joint_travel(astar_plan_wide(note_positions))
    assert abs(ucs_cost - wide_cost) < 1e-6, (
        f"UCS ({ucs_cost:.6f}) and A*-wide ({wide_cost:.6f}) disagree — "
        "one of them has a bug"
    )


def test_ucs_matches_astar_wide_cost_mary():
    note_positions = resolve_melody(MARY)
    ucs_cost  = total_joint_travel(ucs_plan(note_positions))
    wide_cost = total_joint_travel(astar_plan_wide(note_positions))
    assert abs(ucs_cost - wide_cost) < 1e-6, (
        f"UCS ({ucs_cost:.6f}) and A*-wide ({wide_cost:.6f}) disagree on Mary"
    )


def test_ucs_empty_melody():
    assert ucs_plan([]) == []


def test_ucs_raises_on_unreachable():
    with pytest.raises(ValueError):
        ucs_plan([("X99", (100.0, 100.0))])
