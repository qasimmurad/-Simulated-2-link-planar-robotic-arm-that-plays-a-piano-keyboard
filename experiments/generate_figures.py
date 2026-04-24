"""
generate_figures.py

Generates two static figures and saves them to results/figures/.

  workspace_reachability.png  — reachable annulus + all keys colour-coded
                                green (reachable) / red (unreachable)
  twinkle_trajectory.png      — A* plan for Twinkle overlaid on workspace

Run from the project root:
    python experiments/generate_figures.py
"""

import os
import sys

# Headless backend — must be set before any pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.keyboard_layout import WHITE_KEYS, BLACK_KEYS, ALL_KEYS, L1, L2, BASE
from src.robotics.kinematics import is_reachable
from src.robotics.workspace import reachable_keys, unreachable_keys
from src.viz.plots import plot_plan
from src.music.resolver import resolve_melody, TWINKLE
from src.planning.search import astar_plan_wide

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "figures")


def _annulus(ax, base, r_min, r_max):
    theta = np.linspace(0, 2 * np.pi, 500)
    bx, by = base
    ax.fill(
        bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
        alpha=0.08, color="steelblue",
    )
    ax.fill(
        bx + r_min * np.cos(theta), by + r_min * np.sin(theta),
        alpha=1.0, color="white",
    )
    ax.plot(bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
            "b-", lw=1.2, label=f"Outer radius (L1+L2={r_max:.3f} m)")
    ax.plot(bx + r_min * np.cos(theta), by + r_min * np.sin(theta),
            "b--", lw=1.2, label=f"Inner radius (|L1−L2|={r_min:.3f} m)")


def generate_workspace_reachability():
    """Figure 1: workspace annulus + colour-coded key reachability."""
    fig, ax = plt.subplots(figsize=(11, 7))

    bx, by = BASE
    _annulus(ax, BASE, abs(L1 - L2), L1 + L2)

    reach  = reachable_keys()
    unreach = unreachable_keys()

    # Reachable keys — green
    for note, pos in reach.items():
        is_white = note in WHITE_KEYS
        ms = 9 if is_white else 6
        ax.plot(*pos, "o", color="limegreen", markersize=ms,
                markeredgecolor="darkgreen", markeredgewidth=0.8, zorder=4)
        ax.text(pos[0], pos[1] - 0.013, note,
                ha="center", va="top",
                fontsize=7 if is_white else 5.5,
                color="darkgreen")

    # Unreachable keys — red
    for note, pos in unreach.items():
        is_white = note in WHITE_KEYS
        ms = 9 if is_white else 6
        ax.plot(*pos, "o", color="tomato", markersize=ms,
                markeredgecolor="darkred", markeredgewidth=0.8, zorder=4)
        ax.text(pos[0], pos[1] - 0.013, note,
                ha="center", va="top",
                fontsize=7 if is_white else 5.5,
                color="darkred")

    ax.plot(*BASE, "k^", markersize=11, zorder=5, label="Arm base")

    # Legend proxies for reachability colours
    ax.plot([], [], "o", color="limegreen", markeredgecolor="darkgreen",
            label=f"Reachable ({len(reach)} keys)")
    ax.plot([], [], "o", color="tomato",    markeredgecolor="darkred",
            label=f"Unreachable ({len(unreach)} keys)")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Workspace Reachability — 2-link Planar Arm (L1=0.20 m, L2=0.15 m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "workspace_reachability.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_twinkle_trajectory():
    """Figure 2: A* Twinkle plan overlaid on workspace diagram."""
    positions = resolve_melody(TWINKLE)
    plan = astar_plan_wide(positions)

    # plot_plan calls plt.subplots internally; show=False prevents display.
    fig, ax = plot_plan(plan, show=False)
    ax.set_title("A* Planned Trajectory — Twinkle Twinkle (joint-space wide heuristic)")

    path = os.path.join(FIGURES_DIR, "twinkle_trajectory.png")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Generating figures...")
    p1 = generate_workspace_reachability()
    size1 = os.path.getsize(p1)
    print(f"  ✓ {p1}  ({size1:,} bytes)")

    p2 = generate_twinkle_trajectory()
    size2 = os.path.getsize(p2)
    print(f"  ✓ {p2}  ({size2:,} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
