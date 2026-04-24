"""
run_search_comparison.py

Compare greedy, A* (narrow), and A* (wide) motion planning across all melodies.
Run from the project root: python experiments/run_search_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.music.resolver import resolve_melody, MELODIES
from src.planning.search import greedy_plan, astar_plan, astar_plan_wide, total_joint_travel


def main():
    header = (
        f"{'Melody':<14} {'Notes':>5}  {'Greedy':>10}  "
        f"{'A*(narrow)':>12}  {'A*(wide)':>10}  {'Wide saves':>11}"
    )
    print(header)
    print("-" * len(header))

    for name, notes in MELODIES.items():
        positions = resolve_melody(notes)
        gc  = total_joint_travel(greedy_plan(positions))
        ac  = total_joint_travel(astar_plan(positions))
        wc  = total_joint_travel(astar_plan_wide(positions))
        saving = (gc - wc) / gc * 100 if gc > 0 else 0.0
        print(
            f"{name:<14} {len(positions):>5}  {gc:>10.4f}  "
            f"{ac:>12.4f}  {wc:>10.4f}  {saving:>10.1f}%"
        )


if __name__ == "__main__":
    main()
