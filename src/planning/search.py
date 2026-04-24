"""
search.py

Motion planning: find joint-angle sequences to play a melody.

Four strategies + instrumented variants:
  - greedy_plan            : picks the IK solution with least joint travel at each step.
  - astar_plan             : A* over both IK solutions (elbow-up / elbow-down) per step.
  - astar_plan_wide        : A* over 6 approach configs per step (2 base IK × 3 theta1
                             perturbations), giving a wide enough branching factor for
                             A* to beat greedy on joint-travel cost.
  - ucs_plan               : uniform-cost search (Dijkstra) over the same wide state
                             space as astar_plan_wide but with h=0 everywhere.
  - astar_plan_wide_instrumented: like astar_plan_wide but returns PlanResult with
                             plan, nodes_expanded, and runtime_ms.
  - ucs_plan_instrumented  : like ucs_plan but returns PlanResult with the same fields.
"""

import heapq
import time
import numpy as np
from typing import List, NamedTuple, Tuple

from data.keyboard_layout import L1 as _L1, L2 as _L2, BASE as _BASE
from src.robotics.kinematics import inverse_kinematics, choose_solution
from src.planning.heuristics import joint_travel_cost, joint_space_heuristic

_THETA1_DELTAS = (-0.1, 0.0, 0.1)

NotePos = Tuple[str, Tuple[float, float]]
Config = Tuple[float, float]


class PlanResult(NamedTuple):
    """Return type for instrumented search functions."""
    plan: List[Tuple[str, float, float]]
    nodes_expanded: int
    runtime_ms: float


def greedy_plan(note_positions: List[NotePos]) -> List[Tuple[str, float, float]]:
    """
    Greedy nearest-neighbour plan.

    At each step choose the IK solution that minimises joint travel
    from the current configuration.

    Returns list of (note, theta1, theta2).
    Raises ValueError if any note is unreachable.
    """
    plan: List[Tuple[str, float, float]] = []
    t1 = 0.0

    for note, pos in note_positions:
        sol = choose_solution(pos, current_theta1=t1)
        if sol is None:
            raise ValueError(f"Note {note!r} at {pos} is unreachable from current config")
        plan.append((note, sol[0], sol[1]))
        t1 = sol[0]

    return plan


def astar_plan(note_positions: List[NotePos]) -> List[Tuple[str, float, float]]:
    """
    A* search over both IK solutions at each melody step.

    Minimises total joint travel across the whole sequence.
    Returns list of (note, theta1, theta2).
    Raises ValueError if any note is unreachable.
    """
    if not note_positions:
        return []

    n = len(note_positions)
    # heap entry: (f, g, step, t1, t2, path)
    heap = [(0.0, 0.0, 0, 0.0, 0.0, [])]
    # best g-cost seen for (step, rounded angles)
    visited: dict = {}

    while heap:
        _, g, step, t1, t2, path = heapq.heappop(heap)

        if step == n:
            return path

        note, pos = note_positions[step]
        for elbow_up in (True, False):
            sol = inverse_kinematics(pos, elbow_up=elbow_up)
            if sol is None:
                continue

            nt1, nt2 = sol
            new_g = g + joint_travel_cost(t1, t2, nt1, nt2)

            state_key = (step + 1, round(nt1, 5), round(nt2, 5))
            if state_key in visited and visited[state_key] <= new_g:
                continue
            visited[state_key] = new_g

            h = (
                joint_space_heuristic((nt1, nt2), note_positions[step + 1][1])
                if step + 1 < n
                else 0.0
            )
            new_path = path + [(note, nt1, nt2)]
            heapq.heappush(heap, (new_g + h, new_g, step + 1, nt1, nt2, new_path))

    raise ValueError("No valid plan found — check that all notes are reachable")


def total_joint_travel(plan: List[Tuple[str, float, float]]) -> float:
    """Sum of joint-angle changes across a plan."""
    if len(plan) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(plan)):
        _, t1a, t2a = plan[i - 1]
        _, t1b, t2b = plan[i]
        total += joint_travel_cost(t1a, t2a, t1b, t2b)
    return total


# ---------------------------------------------------------------------------
# Wide-state helpers
# ---------------------------------------------------------------------------

def _wide_configs(
    pos: Tuple[float, float],
    base: Tuple[float, float] = _BASE,
    l1: float = _L1,
    l2: float = _L2,
) -> List[Tuple[float, float]]:
    """
    Generate up to 6 candidate (theta1, theta2) configurations for a key at pos.

    For each base IK solution (elbow-up and elbow-down), perturbs theta1 by
    each value in _THETA1_DELTAS and re-solves theta2 so that link 2 points
    along the approach vector from the new elbow position toward the key.
    This widens the branching factor from 2 to up to 6 per melody step.
    """
    bx, by = base
    px, py = pos
    candidates: List[Tuple[float, float]] = []

    for elbow_up in (True, False):
        base_sol = inverse_kinematics(pos, base=base, l1=l1, l2=l2, elbow_up=elbow_up)
        if base_sol is None:
            continue
        theta1_base, _ = base_sol

        for delta in _THETA1_DELTAS:
            new_t1 = theta1_base + delta
            # Perturbed elbow position
            ex = bx + l1 * np.cos(new_t1)
            ey = by + l1 * np.sin(new_t1)
            dx, dy = px - ex, py - ey
            if np.hypot(dx, dy) < 1e-9:
                continue
            # theta2 that points link 2 along the approach vector toward the key
            new_t2 = np.arctan2(dy, dx) - new_t1
            candidates.append((float(new_t1), float(new_t2)))

    return candidates


def _wide_heuristic(
    current_angles: Tuple[float, float],
    target_pos: Tuple[float, float],
) -> float:
    """
    Admissible heuristic for astar_plan_wide.

    Returns the minimum joint travel over all wide candidate configs for
    target_pos. Admissible because it never exceeds the true optimal cost
    to reach any valid configuration for that note.
    """
    configs = _wide_configs(target_pos)
    if not configs:
        return float("inf")
    t1, t2 = current_angles
    return min(joint_travel_cost(t1, t2, nt1, nt2) for nt1, nt2 in configs)


def _run_wide_search(
    note_positions: List[NotePos],
    use_heuristic: bool,
) -> PlanResult:
    """
    Shared engine for astar_plan_wide and ucs_plan.

    When use_heuristic=True the priority is f = g + h (_wide_heuristic).
    When use_heuristic=False, h ≡ 0 so the priority is purely g (Dijkstra).
    Both modes count every heapq.heappop as one expanded node and measure
    wall-clock time with time.perf_counter.
    """
    if not note_positions:
        return PlanResult(plan=[], nodes_expanded=0, runtime_ms=0.0)

    n = len(note_positions)
    nodes_expanded = 0
    t_start = time.perf_counter()

    # Unified heap entry: (f, g, step, t1, t2, path).
    # For UCS, f == g because h ≡ 0; the extra column costs nothing.
    heap = [(0.0, 0.0, 0, 0.0, 0.0, [])]
    visited: dict = {}

    while heap:
        _, g, step, t1, t2, path = heapq.heappop(heap)
        nodes_expanded += 1

        if step == n:
            runtime_ms = (time.perf_counter() - t_start) * 1_000.0
            return PlanResult(plan=path, nodes_expanded=nodes_expanded, runtime_ms=runtime_ms)

        note, pos = note_positions[step]
        for nt1, nt2 in _wide_configs(pos):
            new_g = g + joint_travel_cost(t1, t2, nt1, nt2)

            state_key = (step + 1, round(nt1, 4), round(nt2, 4))
            if state_key in visited and visited[state_key] <= new_g:
                continue
            visited[state_key] = new_g

            h = (
                _wide_heuristic((nt1, nt2), note_positions[step + 1][1])
                if (use_heuristic and step + 1 < n)
                else 0.0
            )
            new_path = path + [(note, nt1, nt2)]
            heapq.heappush(heap, (new_g + h, new_g, step + 1, nt1, nt2, new_path))

    raise ValueError("No valid plan found — check that all notes are reachable")


def astar_plan_wide(note_positions: List[NotePos]) -> List[Tuple[str, float, float]]:
    """
    A* search over a widened set of approach configurations per melody step.

    For each note, considers up to 6 configurations (2 base IK solutions ×
    3 theta1 perturbations of {-0.1, 0, +0.1} rad).  For each perturbation,
    theta2 is re-solved so that link 2 points along the approach vector from
    the perturbed elbow toward the key.  This wider branching lets A* find
    plans with strictly lower total joint travel than greedy_plan.

    Drop-in compatible with astar_plan: same signature and return type.
    For stats (nodes expanded, runtime) use astar_plan_wide_instrumented.
    Raises ValueError if no valid plan exists.
    """
    return _run_wide_search(note_positions, use_heuristic=True).plan


def ucs_plan(note_positions: List[NotePos]) -> List[Tuple[str, float, float]]:
    """
    Uniform-cost search (Dijkstra) over the wide state space.

    Identical to astar_plan_wide except h=0 everywhere.  Both must return
    plans with identical total joint travel; any discrepancy indicates a bug.

    For stats (nodes expanded, runtime) use ucs_plan_instrumented.
    Raises ValueError if no valid plan exists.
    """
    return _run_wide_search(note_positions, use_heuristic=False).plan


def astar_plan_wide_instrumented(note_positions: List[NotePos]) -> PlanResult:
    """
    Like astar_plan_wide but returns a PlanResult with:
      .plan           — list of (note, theta1, theta2)
      .nodes_expanded — number of nodes popped from the priority queue
      .runtime_ms     — wall-clock time in milliseconds

    Use this to measure how much the admissible heuristic reduces search effort
    compared to ucs_plan_instrumented.
    """
    return _run_wide_search(note_positions, use_heuristic=True)


# Backward-compatible alias added in Task 3 under a shorter name.
astar_plan_instrumented = astar_plan_wide_instrumented


def ucs_plan_instrumented(note_positions: List[NotePos]) -> PlanResult:
    """
    Like ucs_plan but returns a PlanResult with:
      .plan           — list of (note, theta1, theta2)
      .nodes_expanded — number of nodes popped from the priority queue
      .runtime_ms     — wall-clock time in milliseconds

    Baseline for comparing against astar_plan_instrumented: same optimal cost,
    more nodes expanded (no heuristic to prune the frontier).
    """
    return _run_wide_search(note_positions, use_heuristic=False)
