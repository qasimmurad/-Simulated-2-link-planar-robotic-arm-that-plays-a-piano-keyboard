"""
kinematics.py

Forward and inverse kinematics for a 2-link planar robotic arm.

Derivation
----------
The arm has two rigid links of lengths L1 and L2, with a base at
position BASE = (bx, by). Joint 1 (shoulder) rotates by theta1,
joint 2 (elbow) rotates by theta2 relative to link 1.

Forward kinematics:
    elbow = BASE + L1 * [cos(theta1), sin(theta1)]
    tip   = elbow + L2 * [cos(theta1 + theta2), sin(theta1 + theta2)]

Inverse kinematics (geometric method):
    Given target (px, py), find (theta1, theta2).

    Step 1: Translate to arm-local coordinates.
        dx = px - bx
        dy = py - by
        r  = sqrt(dx^2 + dy^2)

    Step 2: Check reachability.
        |L1 - L2| <= r <= L1 + L2  (triangle inequality)

    Step 3: Solve theta2 via cosine rule.
        cos(theta2) = (r^2 - L1^2 - L2^2) / (2 * L1 * L2)
        theta2 = +/- arccos(...)   (elbow-up and elbow-down solutions)

    Step 4: Solve theta1.
        alpha = atan2(dy, dx)
        beta  = arccos((r^2 + L1^2 - L2^2) / (2 * r * L1))
        elbow-down: theta1 = alpha - beta
        elbow-up:   theta1 = alpha + beta
"""

import numpy as np
from typing import Optional, Tuple
from data.keyboard_layout import L1, L2, BASE


def forward_kinematics(
    theta1: float,
    theta2: float,
    base: Tuple[float, float] = BASE,
    l1: float = L1,
    l2: float = L2,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Compute elbow and tip positions given joint angles.

    Parameters
    ----------
    theta1 : float
        Shoulder joint angle in radians.
    theta2 : float
        Elbow joint angle in radians (relative to link 1).
    base : tuple
        (x, y) position of the arm base.
    l1, l2 : float
        Link lengths in metres.

    Returns
    -------
    base, elbow, tip : each a (float, float) tuple
    """
    bx, by = base
    elbow_x = bx + l1 * np.cos(theta1)
    elbow_y = by + l1 * np.sin(theta1)
    tip_x = elbow_x + l2 * np.cos(theta1 + theta2)
    tip_y = elbow_y + l2 * np.sin(theta1 + theta2)
    return (bx, by), (elbow_x, elbow_y), (tip_x, tip_y)


def inverse_kinematics(
    target: Tuple[float, float],
    base: Tuple[float, float] = BASE,
    l1: float = L1,
    l2: float = L2,
    elbow_up: bool = True,
) -> Optional[Tuple[float, float]]:
    """
    Compute joint angles to reach a target (x, y) position.

    Parameters
    ----------
    target : (float, float)
        Desired tip position in world coordinates.
    base : (float, float)
        Arm base position.
    l1, l2 : float
        Link lengths in metres.
    elbow_up : bool
        If True, return the elbow-up solution. If False, elbow-down.

    Returns
    -------
    (theta1, theta2) in radians, or None if target is unreachable.
    """
    bx, by = base
    px, py = target
    dx = px - bx
    dy = py - by
    r = np.sqrt(dx**2 + dy**2)

    # Reachability check
    if r > l1 + l2 or r < abs(l1 - l2):
        return None  # outside workspace

    # Cosine rule for theta2
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)  # guard floating point
    theta2 = np.arccos(cos_theta2)
    if not elbow_up:
        theta2 = -theta2

    # Solve theta1
    alpha = np.arctan2(dy, dx)
    beta_cos = (r**2 + l1**2 - l2**2) / (2 * r * l1)
    beta_cos = np.clip(beta_cos, -1.0, 1.0)
    beta = np.arccos(beta_cos)

    if elbow_up:
        theta1 = alpha - beta
    else:
        theta1 = alpha + beta

    return theta1, theta2


def is_reachable(
    target: Tuple[float, float],
    base: Tuple[float, float] = BASE,
    l1: float = L1,
    l2: float = L2,
) -> bool:
    """
    Return True if the target is within the arm's reachable workspace.
    """
    bx, by = base
    r = np.sqrt((target[0] - bx)**2 + (target[1] - by)**2)
    return abs(l1 - l2) <= r <= l1 + l2


def choose_solution(
    target: Tuple[float, float],
    current_theta1: float = 0.0,
    base: Tuple[float, float] = BASE,
    l1: float = L1,
    l2: float = L2,
) -> Optional[Tuple[float, float]]:
    """
    Choose between elbow-up and elbow-down solutions.

    Strategy: pick the solution whose theta1 requires less
    joint travel from the current shoulder angle. This minimises
    unnecessary large joint excursions between consecutive keys.

    Parameters
    ----------
    current_theta1 : float
        Current shoulder angle (radians) before this move.

    Returns
    -------
    (theta1, theta2) for the preferred solution, or None if unreachable.
    """
    up   = inverse_kinematics(target, base, l1, l2, elbow_up=True)
    down = inverse_kinematics(target, base, l1, l2, elbow_up=False)

    if up is None and down is None:
        return None
    if up is None:
        return down
    if down is None:
        return up

    # Pick the solution with smaller theta1 deviation from current pose
    if abs(up[0] - current_theta1) <= abs(down[0] - current_theta1):
        return up
    return down
