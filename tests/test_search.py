"""
test_search.py

Unit tests for motion planning (search and heuristics).
"""

import pytest
from src.music.resolver import resolve_melody, TWINKLE, MARY
from src.planning.search import (
    greedy_plan, astar_plan, astar_plan_wide, ucs_plan,
    astar_plan_wide_instrumented, ucs_plan_instrumented,
    PlanResult, total_joint_travel, _wide_configs,
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


# --- instrumented function tests ---

def test_astar_instrumented_returns_plan_result():
    result = astar_plan_wide_instrumented(resolve_melody(TWINKLE))
    assert isinstance(result, PlanResult)


def test_ucs_instrumented_returns_plan_result():
    result = ucs_plan_instrumented(resolve_melody(TWINKLE))
    assert isinstance(result, PlanResult)


def test_instrumented_plan_field_is_list():
    for fn in (astar_plan_wide_instrumented, ucs_plan_instrumented):
        result = fn(resolve_melody(TWINKLE))
        assert isinstance(result.plan, list)
        assert len(result.plan) == len(resolve_melody(TWINKLE))


def test_instrumented_nodes_expanded_is_positive_int():
    for fn in (astar_plan_wide_instrumented, ucs_plan_instrumented):
        result = fn(resolve_melody(TWINKLE))
        assert isinstance(result.nodes_expanded, int)
        assert result.nodes_expanded > 0


def test_instrumented_runtime_ms_is_nonneg_float():
    for fn in (astar_plan_wide_instrumented, ucs_plan_instrumented):
        result = fn(resolve_melody(TWINKLE))
        assert isinstance(result.runtime_ms, float)
        assert result.runtime_ms >= 0.0


def test_instrumented_cost_matches_uninstrumented():
    """Instrumented wrappers must return the exact same plan cost as the plain functions."""
    pos = resolve_melody(TWINKLE)
    assert abs(
        total_joint_travel(astar_plan_wide_instrumented(pos).plan)
        - total_joint_travel(astar_plan_wide(pos))
    ) < 1e-9
    assert abs(
        total_joint_travel(ucs_plan_instrumented(pos).plan)
        - total_joint_travel(ucs_plan(pos))
    ) < 1e-9


def test_ucs_expands_more_nodes_than_astar():
    """UCS has no heuristic so it must expand at least as many nodes as A*-wide."""
    pos = resolve_melody(TWINKLE)
    astar_nodes = astar_plan_wide_instrumented(pos).nodes_expanded
    ucs_nodes   = ucs_plan_instrumented(pos).nodes_expanded
    assert ucs_nodes >= astar_nodes, (
        f"UCS expanded {ucs_nodes} nodes but A* expanded {astar_nodes} — "
        "heuristic should prune at least as many states"
    )


def test_instrumented_empty_melody():
    for fn in (astar_plan_wide_instrumented, ucs_plan_instrumented):
        result = fn([])
        assert result.plan == []
        assert result.nodes_expanded == 0
        assert result.runtime_ms == 0.0


def test_instrumented_raises_on_unreachable():
    for fn in (astar_plan_wide_instrumented, ucs_plan_instrumented):
        with pytest.raises(ValueError):
            fn([("X99", (100.0, 100.0))])


# --- Task 4: heuristic tests ---

from src.planning.heuristics import (
    joint_space_heuristic, h_euclidean_endeffector, h_combined,
)
from src.planning.search import (
    astar_plan_wide_instrumented_euclidean,
    astar_plan_wide_instrumented_combined,
)


def test_h_euclidean_is_admissible():
    """h_euclidean_endeffector must never exceed the true one-step joint travel."""
    from src.robotics.kinematics import inverse_kinematics
    pos_c4 = ALL_KEYS["C4"]
    pos_g4 = ALL_KEYS["G4"]
    # use IK elbow-up for C4 as the starting config
    start = inverse_kinematics(pos_c4, elbow_up=True)
    assert start is not None
    h = h_euclidean_endeffector(start, pos_g4)
    # true cost: min joint travel over both IK solutions for G4
    from src.planning.heuristics import joint_travel_cost
    costs = []
    for eu in (True, False):
        sol = inverse_kinematics(pos_g4, elbow_up=eu)
        if sol is not None:
            costs.append(joint_travel_cost(start[0], start[1], sol[0], sol[1]))
    true_cost = min(costs)
    assert h <= true_cost + 1e-9, (
        f"h_euclidean ({h:.6f}) exceeds true cost ({true_cost:.6f})"
    )


def test_h_euclidean_zero_at_target():
    """Heuristic should be (near) zero when already at the target position."""
    from src.robotics.kinematics import inverse_kinematics
    pos = ALL_KEYS["E4"]
    sol = inverse_kinematics(pos, elbow_up=True)
    assert sol is not None
    h = h_euclidean_endeffector(sol, pos)
    assert h < 1e-9, f"h_euclidean at target should be ~0, got {h}"


def test_h_combined_ge_components():
    """h_combined must be >= both individual components everywhere."""
    from src.robotics.kinematics import inverse_kinematics
    for note_from, note_to in [("C4", "G4"), ("E4", "B4"), ("F4", "C5")]:
        if note_from not in ALL_KEYS or note_to not in ALL_KEYS:
            continue
        sol = inverse_kinematics(ALL_KEYS[note_from], elbow_up=True)
        if sol is None:
            continue
        pos_to = ALL_KEYS[note_to]
        h_js  = joint_space_heuristic(sol, pos_to)
        h_eu  = h_euclidean_endeffector(sol, pos_to)
        h_comb = h_combined(sol, pos_to)
        assert h_comb >= h_js  - 1e-12
        assert h_comb >= h_eu  - 1e-12


def test_h_combined_is_admissible():
    """h_combined must never exceed the true one-step joint travel."""
    from src.robotics.kinematics import inverse_kinematics
    from src.planning.heuristics import joint_travel_cost
    pos_c4 = ALL_KEYS["C4"]
    pos_g4 = ALL_KEYS["G4"]
    start = inverse_kinematics(pos_c4, elbow_up=True)
    assert start is not None
    h = h_combined(start, pos_g4)
    costs = []
    for eu in (True, False):
        sol = inverse_kinematics(pos_g4, elbow_up=eu)
        if sol is not None:
            costs.append(joint_travel_cost(start[0], start[1], sol[0], sol[1]))
    true_cost = min(costs)
    assert h <= true_cost + 1e-9, (
        f"h_combined ({h:.6f}) exceeds true cost ({true_cost:.6f})"
    )


def test_euclidean_instrumented_returns_plan_result():
    result = astar_plan_wide_instrumented_euclidean(resolve_melody(TWINKLE))
    assert isinstance(result, PlanResult)
    assert isinstance(result.plan, list)
    assert len(result.plan) == len(resolve_melody(TWINKLE))
    assert result.nodes_expanded > 0
    assert result.runtime_ms >= 0.0


def test_combined_instrumented_returns_plan_result():
    result = astar_plan_wide_instrumented_combined(resolve_melody(TWINKLE))
    assert isinstance(result, PlanResult)
    assert isinstance(result.plan, list)
    assert len(result.plan) == len(resolve_melody(TWINKLE))
    assert result.nodes_expanded > 0
    assert result.runtime_ms >= 0.0


def test_euclidean_instrumented_optimal_cost():
    """Euclidean-heuristic A* must find the same optimal cost as UCS."""
    pos = resolve_melody(TWINKLE)
    ucs_cost = total_joint_travel(ucs_plan(pos))
    eu_cost  = total_joint_travel(astar_plan_wide_instrumented_euclidean(pos).plan)
    assert abs(eu_cost - ucs_cost) < 1e-6, (
        f"Euclidean A* ({eu_cost:.6f}) diverges from UCS ({ucs_cost:.6f})"
    )


def test_combined_instrumented_optimal_cost():
    """Combined-heuristic A* must find the same optimal cost as UCS."""
    pos = resolve_melody(TWINKLE)
    ucs_cost   = total_joint_travel(ucs_plan(pos))
    comb_cost  = total_joint_travel(astar_plan_wide_instrumented_combined(pos).plan)
    assert abs(comb_cost - ucs_cost) < 1e-6, (
        f"Combined A* ({comb_cost:.6f}) diverges from UCS ({ucs_cost:.6f})"
    )


def test_combined_expands_fewer_nodes_than_euclidean():
    """Combined heuristic is tighter so should expand <= nodes vs euclidean-only."""
    pos = resolve_melody(TWINKLE)
    eu_nodes   = astar_plan_wide_instrumented_euclidean(pos).nodes_expanded
    comb_nodes = astar_plan_wide_instrumented_combined(pos).nodes_expanded
    assert comb_nodes <= eu_nodes, (
        f"Combined expanded {comb_nodes} nodes but euclidean expanded {eu_nodes} — "
        "combined heuristic should be at least as tight"
    )
