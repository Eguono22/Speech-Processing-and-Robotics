"""
Tests for the localization module (OdometryModel + ParticleFilter).
"""

import math
import numpy as np
import pytest

from localization.odometry import OdometryModel
from localization.particle_filter import ParticleFilter


# ---------------------------------------------------------------------------
# OdometryModel
# ---------------------------------------------------------------------------

class TestOdometryModel:
    def setup_method(self):
        self.model = OdometryModel(alpha1=0.0, alpha2=0.0, alpha3=0.0, alpha4=0.0)

    def test_compute_odometry_forward(self):
        prev = (0.0, 0.0, 0.0)
        curr = (1.0, 0.0, 0.0)
        d_rot1, d_trans, d_rot2 = OdometryModel.compute_odometry(prev, curr)
        assert abs(d_trans - 1.0) < 1e-6
        assert abs(d_rot1) < 1e-6
        assert abs(d_rot2) < 1e-6

    def test_compute_odometry_turn(self):
        prev = (0.0, 0.0, 0.0)
        curr = (0.0, 0.0, math.pi / 2)
        d_rot1, d_trans, d_rot2 = OdometryModel.compute_odometry(prev, curr)
        assert abs(d_trans) < 1e-6

    def test_sample_no_noise_single(self):
        pose = np.array([0.0, 0.0, 0.0])
        delta = (0.0, 1.0, 0.0)
        result = self.model.sample(pose, delta, n_samples=1, rng=np.random.default_rng(0))
        assert result.shape == (1, 3)
        assert abs(result[0, 0] - 1.0) < 1e-6
        assert abs(result[0, 1]) < 1e-6

    def test_sample_multiple_particles(self):
        poses = np.zeros((100, 3))
        delta = (0.0, 1.0, 0.0)
        result = self.model.sample(poses, delta, rng=np.random.default_rng(0))
        assert result.shape == (100, 3)
        # With near-zero noise (sqrt(1e-12) floor) all particles should end
        # very close to x=1.  The tolerance reflects the sqrt(1e-12) ≈ 1e-6
        # noise floor spread over the sine component; a small multiple suffices.
        np.testing.assert_allclose(result[:, 0], 1.0, atol=1e-4)

    def test_sample_with_noise(self):
        rng = np.random.default_rng(42)
        noisy_model = OdometryModel(0.05, 0.05, 0.05, 0.05)
        pose = np.array([0.0, 0.0, 0.0])
        delta = (0.0, 2.0, 0.0)
        result = noisy_model.sample(pose, delta, n_samples=1000, rng=rng)
        # With large additive rotation noise the cosine factor introduces
        # a downward bias on the mean x; check the samples spread around
        # a reasonable forward displacement (between 1.5 and 2.5).
        mean_x = result[:, 0].mean()
        assert 1.5 < mean_x < 2.5


# ---------------------------------------------------------------------------
# ParticleFilter
# ---------------------------------------------------------------------------

class TestParticleFilter:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.pf = ParticleFilter(
            n_particles=200,
            map_bounds=(0.0, 10.0, 0.0, 10.0),
            measurement_noise_std=0.5,
            rng=self.rng,
        )

    def test_initial_particles_shape(self):
        assert self.pf.particles.shape == (200, 3)

    def test_initial_weights_uniform(self):
        np.testing.assert_allclose(self.pf.weights, 1.0 / 200)

    def test_predict_changes_particles(self):
        old = self.pf.particles.copy()
        self.pf.predict((0.0, 1.0, 0.0))
        assert not np.allclose(self.pf.particles, old)

    def test_update_renormalises_weights(self):
        obs = [{"range": 2.0, "bearing": 0.0, "landmark": (2.0, 2.0)}]
        self.pf.update(obs)
        assert abs(self.pf.weights.sum() - 1.0) < 1e-9

    def test_resample_preserves_count(self):
        self.pf.predict((0.0, 1.0, 0.0))
        self.pf.resample()
        assert self.pf.particles.shape[0] == 200

    def test_estimated_pose_shape(self):
        assert self.pf.estimated_pose.shape == (3,)

    def test_pose_covariance_shape(self):
        assert self.pf.pose_covariance.shape == (2, 2)

    def test_effective_sample_size(self):
        ess = self.pf.effective_sample_size
        assert 0 < ess <= 200

    def test_reset_gaussian(self):
        self.pf.reset(initial_pose=(5.0, 5.0, 0.0), spread=0.1)
        mean_x = self.pf.particles[:, 0].mean()
        mean_y = self.pf.particles[:, 1].mean()
        assert abs(mean_x - 5.0) < 0.2
        assert abs(mean_y - 5.0) < 0.2

    def test_step_returns_pose(self):
        pose = self.pf.step((0.0, 0.5, 0.0))
        assert pose.shape == (3,)

    def test_localization_converges(self):
        """PF should converge near true pose after many steps with observations."""
        true_pose = [1.0, 1.0, 0.0]
        self.pf.reset(initial_pose=tuple(true_pose), spread=0.05)
        landmark = (3.0, 1.0)
        for _ in range(20):
            dx = landmark[0] - true_pose[0]
            dy = landmark[1] - true_pose[1]
            obs = [{"range": math.hypot(dx, dy), "bearing": 0.0, "landmark": landmark}]
            true_pose[0] += 0.1
            delta = (0.0, 0.1, 0.0)
            self.pf.step(delta, observations=obs)
        est = self.pf.estimated_pose
        assert abs(est[0] - true_pose[0]) < 0.5
