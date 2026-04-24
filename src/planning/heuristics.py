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
    Heuristic estimate: Euclidean end-effector distance scaled by (L1+L2).

    Admissibility — conditional
    ---------------------------
    The Jacobian bound gives:

        ||Δtip||₂  ≤  (L1+L2)·|Δθ₁|  +  L2·|Δθ₂|
                    ≤  (L1+L2)·(|Δθ₁| + |Δθ₂|)

    Rearranging: |Δθ₁| + |Δθ₂|  ≥  ||Δtip||₂ / (L1+L2)

    This bound holds when the goal configuration places the arm tip EXACTLY
    at target_pos.  It is then a genuine lower bound on joint travel, hence
    admissible for planners that only consider exact IK solutions (astar_plan,
    greedy_plan).

    Inadmissibility in the wide-search context
    ------------------------------------------
    _run_wide_search uses _wide_configs, which includes perturbed
    configurations (theta1 ± 0.1 rad) that POINT link-2 toward target_pos
    but whose tip does NOT land exactly there (because L2 ≠ distance from the
    perturbed elbow to the target in general).  The cost to transition from
    such a perturbed config to an identical config at the next step is zero,
    yet h = d/(L1+L2) > 0 because the perturbed tip is displaced from
    target_pos.  This violates the admissibility condition.

    Counterexample (empirically verified on Ode to Joy):
        E4→E4 (repeated note), perturbed wide config:
        h_est ≈ 0.052,  min_actual_joint_travel = 0.000
    """
    _, _, tip = forward_kinematics(current_angles[0], current_angles[1], base=base, l1=l1, l2=l2)
    d = np.hypot(tip[0] - target_pos[0], tip[1] - target_pos[1])
    return d / (l1 + l2)


def h_combined(current_angles, target_pos) -> float:
    """
    Combined heuristic: max(joint_space_heuristic, h_euclidean_endeffector).

    Admissibility — conditional (same caveat as h_euclidean_endeffector)
    --------------------------------------------------------------------
    When both components are admissible (i.e., for planners using only exact
    IK configurations), max() of two admissible lower bounds is itself an
    admissible lower bound — it is tighter than either component alone without
    the double-counting risk of sum().

    When h_euclidean_endeffector is inadmissible (wide-search context with
    perturbed configs), h_combined inherits that inadmissibility: max() of an
    admissible value and an inadmissible value can still overestimate.

    Why max() and not sum()
    -----------------------
    sum() of two admissible heuristics is NOT generally admissible: h1 ≤ h*
    and h2 ≤ h* do NOT guarantee h1+h2 ≤ h* when both bound the same true
    cost.  max() avoids double-counting by selecting the single best bound.
    """
    return max(
        joint_space_heuristic(current_angles, target_pos),
        h_euclidean_endeffector(current_angles, target_pos),
    )
