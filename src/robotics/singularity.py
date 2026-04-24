"""
singularity.py

Jacobian, singularity detection, and manipulability for the 2-link planar arm.
"""

import numpy as np
from data.keyboard_layout import L1, L2


def jacobian(theta1: float, theta2: float, l1: float = L1, l2: float = L2) -> np.ndarray:
    """
    Geometric Jacobian of the 2-link planar arm tip w.r.t. joint angles.

    Returns the 2×2 matrix:
        [[dx/dθ1, dx/dθ2],
         [dy/dθ1, dy/dθ2]]

    where:
        dx/dθ1 = -L1 sin(θ1) - L2 sin(θ1+θ2)
        dx/dθ2 = -L2 sin(θ1+θ2)
        dy/dθ1 =  L1 cos(θ1) + L2 cos(θ1+θ2)
        dy/dθ2 =  L2 cos(θ1+θ2)

    det(J) = L1·L2·sin(θ2)  (zero at full extension θ2=0 and full fold θ2=±π)
    """
    s12 = np.sin(theta1 + theta2)
    c12 = np.cos(theta1 + theta2)
    return np.array([
        [-l1 * np.sin(theta1) - l2 * s12,  -l2 * s12],
        [ l1 * np.cos(theta1) + l2 * c12,   l2 * c12],
    ])


def is_singular(theta1: float, theta2: float, threshold: float = 1e-3) -> bool:
    """
    Return True if the configuration is at or near a kinematic singularity.

    Singularities occur where det(J) = L1·L2·sin(θ2) ≈ 0:
      - θ2 = 0      : arm fully extended (maximum reach)
      - θ2 = ±π     : arm fully folded (minimum reach)
    """
    J = jacobian(theta1, theta2)
    return bool(abs(np.linalg.det(J)) < threshold)


def manipulability(theta1: float, theta2: float) -> float:
    """
    Yoshikawa manipulability measure: sqrt(det(J @ J.T)).

    Equals |det(J)| = L1·L2·|sin(θ2)| for a square Jacobian.
    Zero at singularities; maximum at θ2 = ±π/2 (arm in L-shape).
    """
    J = jacobian(theta1, theta2)
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))
