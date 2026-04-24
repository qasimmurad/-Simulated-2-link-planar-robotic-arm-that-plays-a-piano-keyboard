"""
run_search_comparison.py

Compare greedy vs. A* motion planning across all built-in melodies.
Run from the project root: python experiments/run_search_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.music.resolver import resolve_melody, MELODIES
from src.planning.search import greedy_plan, astar_plan, total_joint_travel


def main():
    header = f"{'Melody':<14} {'Notes':>5}  {'Greedy (rad)':>12}  {'A* (rad)':>10}  {'Saving':>8}"
    print(header)
    print("-" * len(header))

    for name, notes in MELODIES.items():
        positions = resolve_melody(notes)
        greedy = greedy_plan(positions)
        astar  = astar_plan(positions)
        gc = total_joint_travel(greedy)
        ac = total_joint_travel(astar)
        saving = (gc - ac) / gc * 100 if gc > 0 else 0.0
        print(f"{name:<14} {len(positions):>5}  {gc:>12.4f}  {ac:>10.4f}  {saving:>7.1f}%")


if __name__ == "__main__":
    main()
