"""
heuristics.py

Cost and heuristic functions for joint-space motion planning.
"""

import numpy as np
from data.keyboard_layout import L1 as _L1, L2 as _L2, BASE as _BASE
from src.robotics.kinematics import choose_solution, forward_kinematics


def joint_travel_cost(t1a: float, t2a: float, t1b: float, t2b: float) -> float:
    """L1 joint-space distance between two configurations."""
    return abs(t1b - t1a) + abs(t2b - t2a)


def joint_space_heuristic(current_angles, target_pos) -> float:
    """
    Admissible heuristic: minimum joint travel to reach target_pos.
    Uses choose_solution to pick the cheaper IK solution.
    """
    sol = choose_solution(target_pos, current_theta1=current_angles[0])
    if sol is None:
        return float("inf")
    return joint_travel_cost(current_angles[0], current_angles[1], sol[0], sol[1])


def h_euclidean_endeffector(
    current_angles,
    target_pos,
    base=_BASE,
    l1=_L1,
    l2=_L2,
) -> float:
    """
    Admissible heuristic: Euclidean end-effector distance scaled by (L1+L2).

    Derivation (admissibility proof)
    ---------------------------------
    For a 2-link planar arm the tip displacement satisfies:

        ||Δtip||₂  ≤  ||J₁||₂·|Δθ₁| + ||J₂||₂·|Δθ₂|
                    ≤  (L1+L2)·|Δθ₁|  +  L2·|Δθ₂|
                    ≤  (L1+L2)·(|Δθ₁| + |Δθ₂|)

    where ||J₁||₂ = sqrt(L1²+L2²+2L1L2 cos θ₂) ≤ L1+L2 and ||J₂||₂ = L2.

    Rearranging: |Δθ₁| + |Δθ₂|  ≥  ||Δtip||₂ / (L1+L2)

    So h = d / (L1+L2)  is a lower bound on total joint travel, hence admissible.
    The bound is tight when the arm is fully extended (θ₂=0) and the shoulder
    joint alone drives the tip directly along the straight-line path to target.
    """
    _, _, tip = forward_kinematics(current_angles[0], current_angles[1], base=base, l1=l1, l2=l2)
    d = np.hypot(tip[0] - target_pos[0], tip[1] - target_pos[1])
    return d / (l1 + l2)


def h_combined(current_angles, target_pos) -> float:
    """
    Admissible combined heuristic: max(joint_space_heuristic, h_euclidean_endeffector).

    Admissibility proof
    -------------------
    Both component heuristics are individually admissible (each ≤ true cost).
    The maximum of two values that are both ≤ X is itself ≤ X, so max() preserves
    admissibility while producing a tighter (higher) lower bound than either alone.

    Why max() and not sum()
    -----------------------
    sum() of two admissible heuristics is NOT generally admissible: if h1 ≤ h*
    and h2 ≤ h* independently, h1+h2 can exceed h* when both measure overlapping
    portions of the same true cost. max() avoids this because it selects the
    single best lower bound at each state rather than double-counting.
    """
    return max(
        joint_space_heuristic(current_angles, target_pos),
        h_euclidean_endeffector(current_angles, target_pos),
    )
