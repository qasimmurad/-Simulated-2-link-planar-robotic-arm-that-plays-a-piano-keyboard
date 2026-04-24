"""
run_tempo_analysis.py

Estimate the maximum playable BPM for each melody given a joint velocity limit.

The bottleneck is the single largest joint-angle move in the plan.
At BPM b, each note lasts 60/b seconds; the arm must complete its move in that time.
Max BPM = 60 * MAX_JOINT_VELOCITY / max_move_cost

Run from the project root: python experiments/run_tempo_analysis.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from src.music.resolver import resolve_melody, MELODIES
from src.planning.search import greedy_plan, astar_plan
from src.planning.heuristics import joint_travel_cost

MAX_JOINT_VELOCITY = math.pi  # rad/s (assumed hardware limit)


def max_bpm(plan):
    """Maximum BPM at which the plan is executable under MAX_JOINT_VELOCITY."""
    if len(plan) < 2:
        return float("inf")
    worst = max(
        joint_travel_cost(plan[i][1], plan[i][2], plan[i + 1][1], plan[i + 1][2])
        for i in range(len(plan) - 1)
    )
    if worst == 0:
        return float("inf")
    return 60.0 * MAX_JOINT_VELOCITY / worst


def bottleneck_move(plan):
    """Return 'NoteA->NoteB' for the hardest consecutive pair."""
    if len(plan) < 2:
        return "-"
    worst_cost, worst_pair = 0.0, "-"
    for i in range(len(plan) - 1):
        c = joint_travel_cost(plan[i][1], plan[i][2], plan[i + 1][1], plan[i + 1][2])
        if c > worst_cost:
            worst_cost = c
            worst_pair = f"{plan[i][0]}->{plan[i + 1][0]}"
    return worst_pair


def main():
    header = (
        f"{'Melody':<14} {'Alg':>7}  {'Max BPM':>9}  {'Bottleneck':>16}"
    )
    print(header)
    print("-" * len(header))

    for name, notes in MELODIES.items():
        positions = resolve_melody(notes)
        for label, plan in [("greedy", greedy_plan(positions)), ("A*", astar_plan(positions))]:
            bpm = max_bpm(plan)
            bpm_str = f"{bpm:.1f}" if bpm != float("inf") else "inf"
            neck = bottleneck_move(plan)
            print(f"{name:<14} {label:>7}  {bpm_str:>9}  {neck:>16}")
        print()


if __name__ == "__main__":
    main()
