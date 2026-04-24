"""
generate_figures.py

Generates two static figures and saves them to results/figures/.

  workspace_reachability.png  — 5-octave keyboard with three reachability
                                categories: both IK configs (green), one
                                config only (yellow), unreachable (red)
  twinkle_trajectory.png      — A* plan for Twinkle overlaid on workspace,
                                with arm postures post-processed so the
                                elbow sits above the base-to-tip line

Run from the project root:
    python experiments/generate_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.keyboard_layout import WHITE_KEYS, BLACK_KEYS, ALL_KEYS, L1, L2, BASE
from src.robotics.kinematics import inverse_kinematics, forward_kinematics
from src.viz.plots import plot_plan
from src.music.resolver import resolve_melody, TWINKLE
from src.planning.search import astar_plan_wide, total_joint_travel

FIGURES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "results", "figures"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reachability_category(pos):
    """
    Return 'both', 'one', or 'none' for a key position.

    'both' — both elbow-up and elbow-down IK solutions exist and differ.
    'one'  — exactly one solution exists, or both exist but are identical
             (arm is exactly on the workspace boundary circle).
    'none' — target is outside the reachable annulus.
    """
    up   = inverse_kinematics(pos, elbow_up=True)
    down = inverse_kinematics(pos, elbow_up=False)
    if up is None and down is None:
        return 'none'
    if up is None or down is None:
        return 'one'
    # Both exist — check if they are meaningfully different
    if abs(up[0] - down[0]) < 1e-4 and abs(up[1] - down[1]) < 1e-4:
        return 'one'   # degenerate case: exactly on boundary
    return 'both'


def _visual_config(t1, t2, pos):
    """
    Choose the IK solution whose elbow sits ABOVE the base-to-tip line.

    "Above" is defined by a positive cross-product of the (base→tip) vector
    with the (base→elbow) vector: cross > 0 means the elbow is on the
    left-hand side of the direction of travel, which for arms reaching
    upward to piano keys corresponds to the natural forward-reaching posture.

    For display only — does not affect reported A* cost.
    """
    _, elbow, tip = forward_kinematics(t1, t2)
    bx, by = BASE
    tx, ty = tip
    ex, ey = elbow

    # cross > 0  →  elbow on the "above" side  →  keep current solution
    cross = (tx - bx) * (ey - by) - (ty - by) * (ex - bx)
    if cross >= 0:
        return t1, t2

    # Try the alternate IK solution
    for eu in (True, False):
        alt = inverse_kinematics(pos, elbow_up=eu)
        if alt is not None and abs(alt[0] - t1) > 1e-3:
            return alt[0], alt[1]

    return t1, t2  # fallback: no alternate available


def _draw_annulus(ax):
    """Draw the reachable annulus (fill + boundary circles)."""
    bx, by = BASE
    r_max = L1 + L2
    r_min = abs(L1 - L2)
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.fill(
        bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
        alpha=0.07, color="steelblue",
    )
    ax.fill(
        bx + r_min * np.cos(theta), by + r_min * np.sin(theta),
        alpha=1.0, color="white",
    )
    ax.plot(
        bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
        "b-", lw=1.2, label=f"Outer radius (L1+L2 = {r_max:.3f} m)",
    )
    ax.plot(
        bx + r_min * np.cos(theta), by + r_min * np.sin(theta),
        "b--", lw=1.2, label=f"Inner radius (|L1−L2| = {r_min:.3f} m)",
    )


# ---------------------------------------------------------------------------
# Figure 1: workspace reachability with 3 categories
# ---------------------------------------------------------------------------

def generate_workspace_reachability():
    fig, ax = plt.subplots(figsize=(16, 6))
    _draw_annulus(ax)

    # Classify every key
    cat_style = {
        'both': dict(color="limegreen",  edge="darkgreen",  label_color="darkgreen"),
        'one':  dict(color="gold",       edge="goldenrod",  label_color="darkgoldenrod"),
        'none': dict(color="tomato",     edge="darkred",    label_color="darkred"),
    }
    counts = {'both': 0, 'one': 0, 'none': 0}

    for note, pos in WHITE_KEYS.items():
        cat = _reachability_category(pos)
        counts[cat] += 1
        s = cat_style[cat]
        ax.plot(*pos, "o", color=s['color'], markersize=10,
                markeredgecolor=s['edge'], markeredgewidth=0.8, zorder=4)
        ax.text(pos[0], pos[1] - 0.014, note,
                ha="center", va="top", fontsize=6, color=s['label_color'])

    for note, pos in BLACK_KEYS.items():
        cat = _reachability_category(pos)
        counts[cat] += 1
        s = cat_style[cat]
        ax.plot(*pos, "o", color=s['color'], markersize=6,
                markeredgecolor=s['edge'], markeredgewidth=0.7, zorder=4)

    ax.plot(*BASE, "k^", markersize=12, zorder=5, label="Arm base")

    # Legend entries with counts
    ax.plot([], [], "o", color="limegreen", markeredgecolor="darkgreen", markersize=9,
            label=f"Reachable — both configs: {counts['both']} keys")
    ax.plot([], [], "o", color="gold", markeredgecolor="goldenrod", markersize=9,
            label=f"Reachable — one config only: {counts['one']} keys")
    ax.plot([], [], "o", color="tomato", markeredgecolor="darkred", markersize=9,
            label=f"Unreachable: {counts['none']} keys")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("y (m)", fontsize=10)
    ax.set_title(
        "Workspace Reachability — 5-octave keyboard (C2–C7)\n"
        f"L1 = {L1:.2f} m, L2 = {L2:.2f} m, "
        f"BASE = ({BASE[0]:.4f}, {BASE[1]:.1f}) m",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "workspace_reachability.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: Twinkle trajectory with corrected elbow orientation
# ---------------------------------------------------------------------------

def generate_twinkle_trajectory():
    positions  = resolve_melody(TWINKLE)
    astar      = astar_plan_wide(positions)
    astar_cost = total_joint_travel(astar)

    # Build note→position map (safe for repeated notes: position is the same)
    note_to_pos = {note: pos for note, pos in positions}

    # Post-process plan for display: choose elbow-above-line orientation
    visual_plan = []
    for note, t1, t2 in astar:
        pos = note_to_pos[note]
        vt1, vt2 = _visual_config(t1, t2, pos)
        visual_plan.append((note, vt1, vt2))

    fig, ax = plot_plan(visual_plan, show=False)
    ax.set_title(
        f"A* Planned Trajectory — Twinkle Twinkle\n"
        f"joint-space wide heuristic  |  plan cost = {astar_cost:.4f} rad",
        fontsize=11,
    )

    path = os.path.join(FIGURES_DIR, "twinkle_trajectory.png")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")

    p1 = generate_workspace_reachability()
    print(f"  ✓ {p1}  ({os.path.getsize(p1):,} bytes)")

    p2 = generate_twinkle_trajectory()
    print(f"  ✓ {p2}  ({os.path.getsize(p2):,} bytes)")

    print("Done.")


if __name__ == "__main__":
    main()
