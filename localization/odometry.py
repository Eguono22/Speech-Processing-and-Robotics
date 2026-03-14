"""
Odometry motion model for a differential-drive robot.

The model propagates a robot pose ``(x, y, theta)`` forward using the
standard probabilistic odometry model described in Thrun, Burgard &
Fox – *Probabilistic Robotics* (MIT Press, 2005), Chapter 5.
"""

from __future__ import annotations

import math
import numpy as np


class OdometryModel:
    """Probabilistic odometry motion model.

    Parameters
    ----------
    alpha1 : float
        Noise coefficient for rotation-from-rotation error.
    alpha2 : float
        Noise coefficient for rotation-from-translation error.
    alpha3 : float
        Noise coefficient for translation-from-translation error.
    alpha4 : float
        Noise coefficient for translation-from-rotation error.
    """

    def __init__(
        self,
        alpha1: float = 0.01,
        alpha2: float = 0.01,
        alpha3: float = 0.01,
        alpha4: float = 0.01,
    ) -> None:
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        pose: np.ndarray,
        delta_odom: tuple[float, float, float],
        n_samples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw *n_samples* noisy successor poses from the motion model.

        Parameters
        ----------
        pose : array-like, shape ``(3,)`` or ``(N, 3)``
            Current pose(s) ``[x, y, theta]``.
        delta_odom : tuple ``(d_rot1, d_trans, d_rot2)``
            Odometry increment decoded from wheel encoders.
        n_samples : int
            Number of samples to generate when *pose* is a single pose.
            When *pose* is an array of N poses, one sample per row is
            returned and *n_samples* is ignored.
        rng : numpy.random.Generator, optional
            Random-number generator (reproducibility).

        Returns
        -------
        numpy.ndarray, shape ``(n_samples, 3)`` or ``(N, 3)``
            Propagated poses with noise applied.
        """
        if rng is None:
            rng = np.random.default_rng()

        pose = np.asarray(pose, dtype=float)
        single_pose = pose.ndim == 1
        if single_pose:
            pose = np.tile(pose, (n_samples, 1))

        d_rot1, d_trans, d_rot2 = delta_odom
        N = pose.shape[0]

        # Sample noisy odometry components
        var_rot1 = self.alpha1 * d_rot1 ** 2 + self.alpha2 * d_trans ** 2
        var_trans = self.alpha3 * d_trans ** 2 + self.alpha4 * (d_rot1 ** 2 + d_rot2 ** 2)
        var_rot2 = self.alpha1 * d_rot2 ** 2 + self.alpha2 * d_trans ** 2

        noisy_rot1 = d_rot1 - rng.normal(0.0, math.sqrt(var_rot1 + 1e-12), N)
        noisy_trans = d_trans - rng.normal(0.0, math.sqrt(var_trans + 1e-12), N)
        noisy_rot2 = d_rot2 - rng.normal(0.0, math.sqrt(var_rot2 + 1e-12), N)

        # Apply motion
        new_x = pose[:, 0] + noisy_trans * np.cos(pose[:, 2] + noisy_rot1)
        new_y = pose[:, 1] + noisy_trans * np.sin(pose[:, 2] + noisy_rot1)
        new_theta = _wrap_angle(pose[:, 2] + noisy_rot1 + noisy_rot2)

        return np.column_stack([new_x, new_y, new_theta])

    @staticmethod
    def compute_odometry(
        prev_pose: tuple[float, float, float],
        curr_pose: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Decompose two consecutive *ground-truth* poses into odometry increments.

        Parameters
        ----------
        prev_pose, curr_pose : tuple ``(x, y, theta)``

        Returns
        -------
        tuple ``(d_rot1, d_trans, d_rot2)``
        """
        dx = curr_pose[0] - prev_pose[0]
        dy = curr_pose[1] - prev_pose[1]
        d_trans = math.hypot(dx, dy)
        d_rot1 = _wrap_angle(math.atan2(dy, dx) - prev_pose[2]) if d_trans > 1e-6 else 0.0
        d_rot2 = _wrap_angle(curr_pose[2] - prev_pose[2] - d_rot1)
        return d_rot1, d_trans, d_rot2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap *angle* to the interval ``(-pi, pi]``."""
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi
