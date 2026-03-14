"""
Mathematical utilities shared across the system.
"""

from __future__ import annotations

import math
import numpy as np


def normalize_angle(angle: float) -> float:
    """Return *angle* (radians) mapped to ``[0, 2π)``."""
    return angle % (2 * math.pi)


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap *angle* (radians) to the interval ``(−π, π]``."""
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


def gaussian(x: float | np.ndarray, sigma: float) -> float | np.ndarray:
    """Evaluate a zero-mean Gaussian PDF with standard deviation *sigma*.

    Parameters
    ----------
    x : float or numpy.ndarray
        Argument(s).
    sigma : float
        Standard deviation (must be > 0).

    Returns
    -------
    float or numpy.ndarray
    """
    return np.exp(-0.5 * (np.asarray(x) / sigma) ** 2) / (math.sqrt(2 * math.pi) * sigma)


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """Return a 2-D rotation matrix for angle *theta* (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Return the Euclidean distance between two 2-D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def compute_bearing(from_pose: tuple[float, float, float], to_point: tuple[float, float]) -> float:
    """Compute the bearing (radians) from *from_pose* to *to_point*.

    Parameters
    ----------
    from_pose : tuple ``(x, y, theta)``
    to_point  : tuple ``(x, y)``

    Returns
    -------
    float – bearing in ``(−π, π]``
    """
    dx = to_point[0] - from_pose[0]
    dy = to_point[1] - from_pose[1]
    return float(wrap_to_pi(math.atan2(dy, dx) - from_pose[2]))
