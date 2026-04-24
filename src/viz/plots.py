"""
plots.py

Static matplotlib plots: workspace boundary, key layout, arm configs, plans.
"""

import numpy as np
import matplotlib.pyplot as plt

from data.keyboard_layout import WHITE_KEYS, BLACK_KEYS, L1, L2, BASE
from src.robotics.kinematics import forward_kinematics


def plot_workspace(ax=None, show=True):
    """Draw the reachable annulus and all key positions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    bx, by = BASE
    theta = np.linspace(0, 2 * np.pi, 500)
    r_max = L1 + L2
    r_min = abs(L1 - L2)

    ax.fill(
        bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
        alpha=0.08, color="steelblue",
    )
    ax.fill(
        bx + r_min * np.cos(theta), by + r_min * np.sin(theta),
        alpha=1.0, color="white",
    )
    ax.plot(bx + r_max * np.cos(theta), by + r_max * np.sin(theta), "b-", lw=1, label="Workspace")
    ax.plot(bx + r_min * np.cos(theta), by + r_min * np.sin(theta), "b--", lw=1)
    ax.plot(*BASE, "k^", markersize=10, label="Base")

    for note, pos in WHITE_KEYS.items():
        ax.plot(*pos, "wo", markersize=9, markeredgecolor="black", zorder=3)
        ax.text(pos[0], pos[1] - 0.012, note, ha="center", va="top", fontsize=7)

    for note, pos in BLACK_KEYS.items():
        ax.plot(*pos, "ko", markersize=6, zorder=3)
        ax.text(pos[0], pos[1] - 0.012, note, ha="center", va="top", fontsize=6, color="dimgray")

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Arm Workspace and Keyboard Layout")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_arm(theta1: float, theta2: float, ax=None, color="tab:red", alpha=1.0, label=None):
    """Overlay a single arm configuration on ax."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    base, elbow, tip = forward_kinematics(theta1, theta2)
    xs = [base[0], elbow[0], tip[0]]
    ys = [base[1], elbow[1], tip[1]]
    ax.plot(xs, ys, "o-", color=color, alpha=alpha, lw=2.5, label=label)
    return ax


def plot_plan(plan, show=True):
    """Overlay the full planned trajectory on the workspace diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_workspace(ax=ax, show=False)

    n = len(plan)
    cmap = plt.cm.viridis
    for i, (note, t1, t2) in enumerate(plan):
        frac = i / max(n - 1, 1)
        plot_arm(t1, t2, ax=ax, color=cmap(frac), alpha=0.4 + 0.6 * frac,
                 label=note if i < 5 else None)

    ax.set_title("Planned Arm Trajectory")
    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax
