"""
test_singularity.py

Unit tests for the Jacobian, singularity detection, and manipulability.
"""

import numpy as np
import pytest
from src.robotics.singularity import jacobian, is_singular, manipulability
from data.keyboard_layout import L1, L2


def test_jacobian_shape():
    J = jacobian(0.5, 0.3)
    assert J.shape == (2, 2)


def test_jacobian_dtype():
    J = jacobian(0.5, 0.3)
    assert J.dtype in (np.float64, np.float32)


def test_jacobian_formula_spot_check():
    """Verify a concrete value: theta1=0, theta2=pi/2."""
    t1, t2 = 0.0, np.pi / 2
    J = jacobian(t1, t2)
    # s12 = sin(pi/2)=1, c12 = cos(pi/2)=0
    expected = np.array([
        [-L1 * 0.0 - L2 * 1.0,  -L2 * 1.0],
        [ L1 * 1.0 + L2 * 0.0,   L2 * 0.0],
    ])
    np.testing.assert_allclose(J, expected, atol=1e-12)


def test_jacobian_det_equals_l1l2_sin_theta2():
    """det(J) should equal L1·L2·sin(θ2) for all configurations."""
    for t1, t2 in [(0.3, 0.7), (1.2, -0.5), (0.0, np.pi / 3), (-0.8, 2.1)]:
        J = jacobian(t1, t2)
        expected_det = L1 * L2 * np.sin(t2)
        assert abs(np.linalg.det(J) - expected_det) < 1e-12, (
            f"det(J)={np.linalg.det(J):.9f}, expected {expected_det:.9f} "
            f"at theta1={t1}, theta2={t2}"
        )


def test_singular_at_full_extension():
    """theta2=0 → arm fully extended → singular."""
    assert is_singular(0.0, 0.0) is True


def test_singular_at_full_fold():
    """theta2=pi → arm fully folded → singular."""
    assert is_singular(0.5, np.pi) is True


def test_singular_at_negative_full_fold():
    """theta2=-pi → singular (same fold, other side)."""
    assert is_singular(0.5, -np.pi) is True


def test_not_singular_at_neutral():
    """theta1=0, theta2=pi/2 → L-shape → well away from singularity."""
    assert is_singular(0.0, np.pi / 2) is False


def test_not_singular_at_various_configs():
    for t1, t2 in [(0.5, 0.8), (-0.3, 1.2), (1.0, np.pi / 3)]:
        assert is_singular(t1, t2) is False, f"Falsely singular at ({t1}, {t2})"


def test_manipulability_zero_at_singularity():
    """At full extension (theta2=0) manipulability must be zero."""
    assert manipulability(0.0, 0.0) == pytest.approx(0.0, abs=1e-10)


def test_manipulability_positive_elsewhere():
    """Away from singularities manipulability must be strictly positive."""
    assert manipulability(0.0, np.pi / 2) > 0.0


def test_manipulability_equals_abs_det_J():
    """For a 2×2 Jacobian, manipulability = |det(J)|."""
    for t1, t2 in [(0.4, 0.9), (1.1, -0.6), (0.0, np.pi / 4)]:
        J = jacobian(t1, t2)
        expected = abs(np.linalg.det(J))
        assert manipulability(t1, t2) == pytest.approx(expected, abs=1e-12)


def test_manipulability_max_at_l_shape():
    """Manipulability peaks at theta2=pi/2 where sin(theta2)=1."""
    m_lshape = manipulability(0.0, np.pi / 2)
    m_extended = manipulability(0.0, 0.0)
    assert m_lshape > m_extended


def test_manipulability_nonnegative():
    """manipulability must never be negative."""
    for t1, t2 in [(0.0, 0.0), (0.5, np.pi), (1.0, 0.7), (-0.3, -1.2)]:
        assert manipulability(t1, t2) >= 0.0
