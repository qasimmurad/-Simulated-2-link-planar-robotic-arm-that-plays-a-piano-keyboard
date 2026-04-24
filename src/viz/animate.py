"""
animate.py

Animated visualisation of the arm playing a melody.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from data.keyboard_layout import WHITE_KEYS, BLACK_KEYS, L1, L2, BASE
from src.robotics.kinematics import forward_kinematics


def _interpolate(t1a, t2a, t1b, t2b, steps=20):
    """Linearly interpolate between two joint configurations."""
    return [
        (t1a + (t1b - t1a) * i / steps, t2a + (t2b - t2a) * i / steps)
        for i in range(steps + 1)
    ]


def _build_frames(plan, steps_per_move=20):
    """Convert a plan into a flat list of (theta1, theta2, note) frames."""
    frames = []
    t1, t2 = 0.0, 0.0
    for note, nt1, nt2 in plan:
        for ct1, ct2 in _interpolate(t1, t2, nt1, nt2, steps_per_move):
            frames.append((ct1, ct2, note))
        t1, t2 = nt1, nt2
    return frames


def animate_plan(plan, steps_per_move=20, interval=40, save_path=None):
    """
    Animate the arm executing a motion plan.

    Parameters
    ----------
    plan : list of (note, theta1, theta2)
    steps_per_move : int
        Interpolation steps between consecutive poses.
    interval : int
        Milliseconds between animation frames.
    save_path : str or None
        Save to file (.gif requires Pillow, .mp4 requires ffmpeg).
        If None, display interactively.
    """
    frames = _build_frames(plan, steps_per_move)
    if not frames:
        print("Empty plan — nothing to animate.")
        return None

    bx, by = BASE
    r_max = L1 + L2
    theta = np.linspace(0, 2 * np.pi, 300)

    fig, ax = plt.subplots(figsize=(10, 8))

    def _draw_static():
        ax.fill(
            bx + r_max * np.cos(theta), by + r_max * np.sin(theta),
            alpha=0.07, color="steelblue",
        )
        ax.plot(bx + r_max * np.cos(theta), by + r_max * np.sin(theta), "b--", lw=1, alpha=0.4)
        for pos in WHITE_KEYS.values():
            ax.plot(*pos, "wo", markersize=9, markeredgecolor="black", zorder=2)
        for pos in BLACK_KEYS.values():
            ax.plot(*pos, "ko", markersize=6, zorder=2)
        ax.plot(*BASE, "k^", markersize=10, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(bx - r_max - 0.02, bx + r_max + 0.02)
        ax.set_ylim(by - 0.05, by + r_max + 0.02)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("2-Link Arm Playing Piano")
        ax.grid(True, alpha=0.2)

    _draw_static()

    (arm_line,) = ax.plot([], [], "o-", color="crimson", lw=3, markersize=8, zorder=5)
    note_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        fontsize=14, verticalalignment="top", fontweight="bold",
    )

    def init():
        arm_line.set_data([], [])
        note_text.set_text("")
        return arm_line, note_text

    def update(idx):
        ct1, ct2, note = frames[idx]
        base, elbow, tip = forward_kinematics(ct1, ct2)
        arm_line.set_data(
            [base[0], elbow[0], tip[0]],
            [base[1], elbow[1], tip[1]],
        )
        note_text.set_text(f"Note: {note}")
        return arm_line, note_text

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True, interval=interval,
    )

    if save_path:
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        ani.save(save_path, writer=writer)
        print(f"Animation saved to {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return ani
