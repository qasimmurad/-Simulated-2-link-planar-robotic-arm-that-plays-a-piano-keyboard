"""
heuristics.py

Cost and heuristic functions for joint-space motion planning.
"""

from src.robotics.kinematics import choose_solution


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
