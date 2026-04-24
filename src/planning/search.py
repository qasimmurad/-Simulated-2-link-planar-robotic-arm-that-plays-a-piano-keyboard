"""
search.py

Motion planning: find joint-angle sequences to play a melody.

Two strategies:
  - greedy_plan  : picks the IK solution with least joint travel at each step.
  - astar_plan   : A* over both IK solutions at each step; globally optimal
                   joint-travel under the single-step heuristic.
"""

import heapq
from typing import List, Tuple

from src.robotics.kinematics import inverse_kinematics, choose_solution
from src.planning.heuristics import joint_travel_cost, joint_space_heuristic

NotePos = Tuple[str, Tuple[float, float]]
Config = Tuple[float, float]


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
