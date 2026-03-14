"""
Particle Filter (Monte Carlo Localization) for a mobile robot.

Implements the *SIR* (Sequential Importance Resampling) particle filter
as described in Thrun, Burgard & Fox – *Probabilistic Robotics* (MIT
Press, 2005), Chapter 4.

The filter maintains a set of weighted particles, each representing a
hypothesis about the robot's pose ``(x, y, theta)``.  It supports:

* **Motion update** – propagates particles through the odometry model.
* **Measurement update** – weights particles using a landmark/range sensor
  likelihood function.
* **Resampling** – low-variance (systematic) resampling.
* **Pose estimate** – weighted mean and covariance of the particle cloud.
"""

from __future__ import annotations

import math
import numpy as np

from .odometry import OdometryModel
from utils.math_utils import gaussian


class ParticleFilter:
    """SIR particle filter for 2-D robot pose estimation.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    map_bounds : tuple ``(x_min, x_max, y_min, y_max)``
        Spatial extent used to initialise particles uniformly.
    odometry_model : OdometryModel, optional
        Motion model.  A default model is created when omitted.
    measurement_noise_std : float
        Standard deviation of the range-sensor measurement noise (metres).
    rng : numpy.random.Generator, optional
        Random-number generator (reproducibility).
    """

    def __init__(
        self,
        n_particles: int = 500,
        map_bounds: tuple[float, float, float, float] = (0.0, 10.0, 0.0, 10.0),
        odometry_model: OdometryModel | None = None,
        measurement_noise_std: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_particles = n_particles
        self.map_bounds = map_bounds
        self.odometry_model = odometry_model or OdometryModel()
        self.measurement_noise_std = measurement_noise_std
        self.rng = rng or np.random.default_rng()

        # Initialise particles uniformly within map bounds
        self.particles = self._uniform_init()
        self.weights = np.ones(n_particles) / n_particles

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _uniform_init(self) -> np.ndarray:
        x_min, x_max, y_min, y_max = self.map_bounds
        x = self.rng.uniform(x_min, x_max, self.n_particles)
        y = self.rng.uniform(y_min, y_max, self.n_particles)
        theta = self.rng.uniform(-math.pi, math.pi, self.n_particles)
        return np.column_stack([x, y, theta])

    def reset(self, initial_pose: tuple[float, float, float] | None = None, spread: float = 0.5) -> None:
        """Re-initialise the filter.

        Parameters
        ----------
        initial_pose : tuple ``(x, y, theta)``, optional
            If given, particles are drawn from a Gaussian centred here.
            Otherwise a uniform distribution over ``map_bounds`` is used.
        spread : float
            Standard deviation for Gaussian initialisation (metres /
            radians).
        """
        if initial_pose is None:
            self.particles = self._uniform_init()
        else:
            x0, y0, t0 = initial_pose
            x = self.rng.normal(x0, spread, self.n_particles)
            y = self.rng.normal(y0, spread, self.n_particles)
            theta = self.rng.normal(t0, spread / 2, self.n_particles)
            self.particles = np.column_stack([x, y, theta])
        self.weights = np.ones(self.n_particles) / self.n_particles

    # ------------------------------------------------------------------
    # Core filter steps
    # ------------------------------------------------------------------

    def predict(self, delta_odom: tuple[float, float, float]) -> None:
        """Motion update – propagate all particles through the odometry model.

        Parameters
        ----------
        delta_odom : tuple ``(d_rot1, d_trans, d_rot2)``
            Odometry increment for the current time step.
        """
        self.particles = self.odometry_model.sample(
            self.particles, delta_odom, rng=self.rng
        )

    def update(self, observations: list[dict]) -> None:
        """Measurement update – reweight particles using sensor observations.

        Each observation is a dict with keys:

        * ``"range"`` – measured distance to a landmark (metres).
        * ``"bearing"`` – measured bearing to a landmark (radians).
        * ``"landmark"`` – tuple ``(lx, ly)`` of the known landmark position.

        Parameters
        ----------
        observations : list of dict
        """
        if not observations:
            return

        log_weights = np.zeros(self.n_particles)
        for obs in observations:
            lx, ly = obs["landmark"]
            dx = lx - self.particles[:, 0]
            dy = ly - self.particles[:, 1]
            expected_range = np.hypot(dx, dy)
            log_weights += np.log(
                gaussian(obs["range"] - expected_range, self.measurement_noise_std) + 1e-300
            )

        # Stabilise numerically
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()

    def resample(self) -> None:
        """Low-variance (systematic) resampling."""
        N = self.n_particles
        positions = (self.rng.uniform(0, 1) + np.arange(N)) / N
        cumsum = np.cumsum(self.weights)
        indices = np.searchsorted(cumsum, positions)
        self.particles = self.particles[indices]
        self.weights = np.ones(N) / N

    # ------------------------------------------------------------------
    # Pose estimate
    # ------------------------------------------------------------------

    @property
    def estimated_pose(self) -> np.ndarray:
        """Weighted-mean pose estimate ``[x, y, theta]``."""
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        # Circular mean for angle
        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        theta = math.atan2(sin_mean, cos_mean)
        return np.array([x, y, theta])

    @property
    def pose_covariance(self) -> np.ndarray:
        """Weighted covariance matrix of the ``(x, y)`` components (2×2)."""
        xy = self.particles[:, :2]
        mean_xy = np.average(xy, weights=self.weights, axis=0)
        diff = xy - mean_xy
        cov = np.einsum("n,ni,nj->ij", self.weights, diff, diff)
        return cov

    @property
    def effective_sample_size(self) -> float:
        """Effective number of particles (Neff)."""
        return 1.0 / np.sum(self.weights ** 2)

    # ------------------------------------------------------------------
    # Convenience: full step
    # ------------------------------------------------------------------

    def step(
        self,
        delta_odom: tuple[float, float, float],
        observations: list[dict] | None = None,
        resample_threshold: float = 0.5,
    ) -> np.ndarray:
        """Run one full predict-update-resample cycle.

        Parameters
        ----------
        delta_odom : tuple
            Odometry increment.
        observations : list of dict, optional
            Sensor observations for the measurement update.
        resample_threshold : float
            Fraction of *n_particles* below which resampling is triggered
            (0 = never, 1 = always).

        Returns
        -------
        numpy.ndarray, shape ``(3,)``
            Estimated pose after the update.
        """
        self.predict(delta_odom)
        if observations:
            self.update(observations)
        if self.effective_sample_size < resample_threshold * self.n_particles:
            self.resample()
        return self.estimated_pose
