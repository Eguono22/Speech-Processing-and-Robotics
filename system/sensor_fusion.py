"""
Sensor fusion utilities.

Combines pose estimates from the particle filter with spatial context
information to produce an enriched *state estimate* and, optionally,
to improve the localization accuracy via context-aware map constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from utils.math_utils import wrap_to_pi


@dataclass
class SystemState:
    """Snapshot of the simultaneous system state at one time step.

    Attributes
    ----------
    pose : numpy.ndarray, shape ``(3,)``
        Robot pose ``[x, y, theta]``.
    pose_covariance : numpy.ndarray, shape ``(2, 2)``
        Covariance of the ``(x, y)`` position estimate.
    context : str
        Predicted spatial context label.
    context_probs : dict[str, float]
        Posterior probability distribution over context labels.
    step : int
        Discrete time step index.
    """

    pose: np.ndarray
    pose_covariance: np.ndarray
    context: str
    context_probs: dict[str, float]
    step: int = 0
    effective_n_particles: float = 0.0
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        x, y, t = self.pose
        return (
            f"Step {self.step:4d} | "
            f"Pose: ({x:+6.3f}, {y:+6.3f}, {np.degrees(t):+6.1f}°) | "
            f"Context: {self.context:12s} | "
            f"Neff: {self.effective_n_particles:.0f}"
        )


class SensorFusion:
    """Fuse particle-filter pose estimates with spatial context information.

    This class maintains a sliding history of system states and provides
    methods to refine estimates using the spatial context as a soft
    constraint.  For example, a *corridor* context implies that the robot
    is likely moving along a narrow axis; *outdoor* implies fewer walls
    and therefore larger measurement uncertainty.

    Parameters
    ----------
    context_uncertainty_map : dict[str, float], optional
        Mapping from context label to a multiplicative factor applied to
        the measurement-noise standard deviation of the particle filter.
        A factor > 1 increases uncertainty (e.g. *outdoor* with fewer
        range-sensor returns).
    history_length : int
        Number of past states to keep in memory.
    """

    # Default context → measurement noise scale factors
    _DEFAULT_UNCERTAINTY: dict[str, float] = {
        "corridor": 0.8,
        "room": 1.0,
        "open_space": 1.5,
        "staircase": 1.2,
        "outdoor": 2.0,
    }

    def __init__(
        self,
        context_uncertainty_map: dict[str, float] | None = None,
        history_length: int = 200,
    ) -> None:
        self.context_uncertainty_map: dict[str, float] = (
            context_uncertainty_map or dict(self._DEFAULT_UNCERTAINTY)
        )
        self.history_length = history_length
        self._history: list[SystemState] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(
        self,
        pose: np.ndarray,
        pose_covariance: np.ndarray,
        context: str,
        context_probs: dict[str, float],
        step: int,
        effective_n_particles: float = 0.0,
    ) -> SystemState:
        """Create a :class:`SystemState` and append it to history.

        Parameters
        ----------
        pose : numpy.ndarray, shape ``(3,)``
        pose_covariance : numpy.ndarray, shape ``(2, 2)``
        context : str
        context_probs : dict
        step : int
        effective_n_particles : float

        Returns
        -------
        SystemState
        """
        state = SystemState(
            pose=pose.copy(),
            pose_covariance=pose_covariance.copy(),
            context=context,
            context_probs=dict(context_probs),
            step=step,
            effective_n_particles=effective_n_particles,
        )
        self._history.append(state)
        if len(self._history) > self.history_length:
            self._history.pop(0)
        return state

    def measurement_noise_scale(self, context: str) -> float:
        """Return the noise-scale factor for a given context label."""
        return self.context_uncertainty_map.get(context, 1.0)

    @property
    def history(self) -> list[SystemState]:
        """Read-only view of the state history."""
        return list(self._history)

    def pose_trajectory(self) -> np.ndarray:
        """Return the historical pose trajectory as an ``(T, 3)`` array."""
        if not self._history:
            return np.empty((0, 3))
        return np.array([s.pose for s in self._history])

    def context_sequence(self) -> list[str]:
        """Return the sequence of context labels from history."""
        return [s.context for s in self._history]

    def context_probability_sequence(self) -> list[dict[str, float]]:
        """Return the sequence of context probability dicts from history."""
        return [s.context_probs for s in self._history]

    def smooth_pose(self, window: int = 5) -> np.ndarray:
        """Apply a simple moving-average smoother to the recorded trajectory.

        Parameters
        ----------
        window : int
            Smoothing window size.

        Returns
        -------
        numpy.ndarray, shape ``(T, 3)``
            Smoothed trajectory (angles averaged circularly).
        """
        traj = self.pose_trajectory()
        if traj.shape[0] < 2:
            return traj

        smoothed = np.zeros_like(traj)
        for t in range(len(traj)):
            lo = max(0, t - window // 2)
            hi = min(len(traj), t + window // 2 + 1)
            chunk = traj[lo:hi]
            smoothed[t, 0] = chunk[:, 0].mean()
            smoothed[t, 1] = chunk[:, 1].mean()
            smoothed[t, 2] = float(
                np.arctan2(np.sin(chunk[:, 2]).mean(), np.cos(chunk[:, 2]).mean())
            )
        return smoothed
