"""
Tests for the system integration module
(SensorFusion + SimultaneousSystem).
"""

import math
import numpy as np
import pytest

from localization.particle_filter import ParticleFilter
from spatial_context.audio_processor import AudioProcessor
from spatial_context.feature_extraction import FeatureExtractor
from spatial_context.context_classifier import ContextClassifier, CONTEXT_LABELS
from system.sensor_fusion import SensorFusion, SystemState
from system.simultaneous_system import SimultaneousSystem


# ---------------------------------------------------------------------------
# Helper: build a trained SimultaneousSystem
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000
AUDIO_DURATION = 0.25  # seconds – short for test speed


def _make_audio(context: str, rng: np.random.Generator) -> np.ndarray:
    n = int(AUDIO_DURATION * SAMPLE_RATE)
    t = np.linspace(0, AUDIO_DURATION, n, endpoint=False)
    freq_map = {
        "corridor": 120, "room": 300, "open_space": 80,
        "staircase": 500, "outdoor": 200,
    }
    freq = freq_map.get(context, 300)
    sig = rng.standard_normal(n) * 0.5 + 0.4 * np.sin(2 * np.pi * freq * t)
    return sig / (np.abs(sig).max() + 1e-10)


def _build_system(n_train_per_class: int = 10) -> SimultaneousSystem:
    rng = np.random.default_rng(0)
    audio_proc = AudioProcessor(sample_rate=SAMPLE_RATE)
    feat_ext = FeatureExtractor(sample_rate=SAMPLE_RATE, n_fft=512)

    # Train classifier on minimal synthetic data
    X, y = [], []
    for lbl in CONTEXT_LABELS:
        for _ in range(n_train_per_class):
            sig = _make_audio(lbl, rng)
            frames = audio_proc.process(sig)
            ps = audio_proc.compute_power_spectrum(frames)
            X.append(feat_ext.extract(ps))
            y.append(lbl)
    clf = ContextClassifier().fit(np.array(X), y)

    pf = ParticleFilter(
        n_particles=100,
        map_bounds=(0.0, 10.0, 0.0, 10.0),
        measurement_noise_std=0.5,
        rng=np.random.default_rng(1),
    )
    return SimultaneousSystem(
        particle_filter=pf,
        audio_processor=audio_proc,
        feature_extractor=feat_ext,
        context_classifier=clf,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# SensorFusion
# ---------------------------------------------------------------------------

class TestSensorFusion:
    def setup_method(self):
        self.fusion = SensorFusion()

    def _dummy_state(self, step=0):
        return self.fusion.fuse(
            pose=np.array([1.0, 2.0, 0.5]),
            pose_covariance=np.eye(2),
            context="room",
            context_probs={"room": 0.9, "corridor": 0.1},
            step=step,
        )

    def test_fuse_returns_system_state(self):
        state = self._dummy_state()
        assert isinstance(state, SystemState)

    def test_history_grows(self):
        for i in range(5):
            self._dummy_state(step=i)
        assert len(self.fusion.history) == 5

    def test_history_capped(self):
        fusion = SensorFusion(history_length=3)
        for i in range(10):
            fusion.fuse(np.zeros(3), np.eye(2), "room", {}, step=i)
        assert len(fusion.history) <= 3

    def test_pose_trajectory_shape(self):
        for i in range(4):
            self._dummy_state(step=i)
        traj = self.fusion.pose_trajectory()
        assert traj.shape == (4, 3)

    def test_context_sequence(self):
        for i in range(3):
            self._dummy_state(step=i)
        seq = self.fusion.context_sequence()
        assert seq == ["room", "room", "room"]

    def test_measurement_noise_scale(self):
        assert self.fusion.measurement_noise_scale("corridor") < 1.0
        assert self.fusion.measurement_noise_scale("outdoor") > 1.0
        assert self.fusion.measurement_noise_scale("unknown_label") == 1.0

    def test_smooth_pose_shape(self):
        for i in range(10):
            self.fusion.fuse(
                np.array([float(i), 0.0, 0.0]), np.eye(2), "room", {}, step=i
            )
        smoothed = self.fusion.smooth_pose(window=3)
        assert smoothed.shape == (10, 3)

    def test_smooth_pose_is_smoother(self):
        """Smoothed x should be closer to linear than raw."""
        for i in range(20):
            noise = np.array([np.random.default_rng(i).normal(0, 0.5), 0.0, 0.0])
            self.fusion.fuse(
                np.array([float(i), 0.0, 0.0]) + noise, np.eye(2), "room", {}, step=i
            )
        raw = self.fusion.pose_trajectory()[:, 0]
        sm = self.fusion.smooth_pose()[:, 0]
        expected = np.arange(20, dtype=float)
        assert np.mean((sm - expected) ** 2) <= np.mean((raw - expected) ** 2) + 0.2


# ---------------------------------------------------------------------------
# SimultaneousSystem
# ---------------------------------------------------------------------------

class TestSimultaneousSystem:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.system = _build_system()

    def _audio(self):
        return _make_audio("room", self.rng)

    def test_step_returns_system_state(self):
        state = self.system.step((0.0, 0.1, 0.0), self._audio())
        assert isinstance(state, SystemState)

    def test_step_increments_counter(self):
        for _ in range(3):
            self.system.step((0.0, 0.1, 0.0), self._audio())
        assert self.system._step == 3

    def test_current_pose_shape(self):
        self.system.step((0.0, 0.1, 0.0), self._audio())
        assert self.system.current_pose.shape == (3,)

    def test_current_context_not_none_after_step(self):
        self.system.step((0.0, 0.1, 0.0), self._audio())
        assert self.system.current_context is not None
        assert self.system.current_context in CONTEXT_LABELS

    def test_state_history_grows(self):
        for _ in range(5):
            self.system.step((0.0, 0.1, 0.0), self._audio())
        assert len(self.system.state_history) == 5

    def test_run_batch(self):
        odom = [(0.0, 0.1, 0.0)] * 10
        audio = [_make_audio("corridor", self.rng) for _ in range(10)]
        states = self.system.run(odom, audio)
        assert len(states) == 10

    def test_run_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            self.system.run([(0.0, 0.1, 0.0)] * 5, [self._audio()] * 3)

    def test_with_observations(self):
        obs = [{"range": 3.0, "bearing": 0.0, "landmark": (3.0, 0.0)}]
        state = self.system.step((0.0, 0.2, 0.0), self._audio(), observations=obs)
        assert isinstance(state, SystemState)
