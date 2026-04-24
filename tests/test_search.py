"""
test_search.py

Unit tests for motion planning (search and heuristics).
"""

import pytest
from src.music.resolver import resolve_melody, TWINKLE, MARY
from src.planning.search import greedy_plan, astar_plan, total_joint_travel
from src.planning.heuristics import joint_travel_cost


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
