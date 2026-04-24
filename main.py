"""
main.py

Entry point: plan and animate the arm playing a melody.

Usage:
    python main.py                        # animate Twinkle Twinkle (A* plan)
    python main.py --melody mary          # different melody
    python main.py --planner greedy       # greedy instead of A*
    python main.py --no-animate           # print plan only (no GUI)
    python main.py --save out.gif         # save animation to file
"""

import argparse

from src.music.resolver import resolve_melody, MELODIES
from src.planning.search import greedy_plan, astar_plan, total_joint_travel
from src.viz.plots import plot_workspace, plot_plan
from src.viz.animate import animate_plan


def parse_args():
    p = argparse.ArgumentParser(description="2-link planar arm piano simulation")
    p.add_argument("--melody",   default="twinkle", choices=list(MELODIES), help="Melody to play")
    p.add_argument("--planner",  default="astar",   choices=["greedy", "astar"], help="Planning algorithm")
    p.add_argument("--no-animate", action="store_true", help="Skip animation, print plan only")
    p.add_argument("--workspace",  action="store_true", help="Show workspace plot before animating")
    p.add_argument("--save",     default=None, metavar="FILE", help="Save animation to .gif or .mp4")
    return p.parse_args()


def main():
    args = parse_args()

    notes = MELODIES[args.melody]
    positions = resolve_melody(notes)

    if not positions:
        print(f"No recognised notes in melody '{args.melody}'.")
        return

    print(f"Melody : {args.melody}  ({len(positions)} notes)")
    print(f"Planner: {args.planner}")

    if args.planner == "astar":
        plan = astar_plan(positions)
    else:
        plan = greedy_plan(positions)

    cost = total_joint_travel(plan)
    print(f"Total joint travel: {cost:.4f} rad\n")
    print(f"{'Step':<5} {'Note':<6}  {'theta1 (rad)':>13}  {'theta2 (rad)':>13}")
    print("-" * 42)
    for i, (note, t1, t2) in enumerate(plan):
        print(f"{i:<5} {note:<6}  {t1:>13.4f}  {t2:>13.4f}")

    if args.workspace:
        plot_workspace()

    if args.no_animate:
        return

    if len(plan) > 1:
        print("\nAnimating... (close the window to exit)")
        animate_plan(plan, save_path=args.save)
    else:
        print("Only one note — showing static arm configuration.")
        plot_plan(plan)


if __name__ == "__main__":
    main()
