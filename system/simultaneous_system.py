"""
Simultaneous mobile robot localization and spatial context recognition system.

This module ties together the particle-filter localization pipeline and the
acoustic spatial-context recognition pipeline.  At each time step the system:

1. Accepts an odometry increment and (optionally) range/bearing observations
   from the robot's sensors.
2. Runs the particle filter predict-update-resample cycle.
3. Accepts a short audio frame from the robot's microphone(s).
4. Extracts acoustic features and classifies the spatial context.
5. Fuses both streams into a :class:`~system.sensor_fusion.SystemState`.

Usage example
-------------
See :mod:`main` for an end-to-end demonstration.
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from localization.particle_filter import ParticleFilter
from localization.odometry import OdometryModel
from spatial_context.audio_processor import AudioProcessor
from spatial_context.feature_extraction import FeatureExtractor
from spatial_context.context_classifier import ContextClassifier
from system.sensor_fusion import SensorFusion, SystemState


class SimultaneousSystem:
    """Orchestrates simultaneous localization and spatial context recognition.

    Parameters
    ----------
    particle_filter : ParticleFilter
        Configured particle filter instance.
    audio_processor : AudioProcessor
        Pre-processing pipeline for raw audio.
    feature_extractor : FeatureExtractor
        Acoustic feature extractor.
    context_classifier : ContextClassifier
        Trained spatial context classifier.
    sensor_fusion : SensorFusion, optional
        Fusion and history manager.
    resample_threshold : float
        Neff/N threshold below which the PF resamples (default: 0.5).
    verbose : bool
        Print state summary at each step (default: False).
    """

    def __init__(
        self,
        particle_filter: ParticleFilter,
        audio_processor: AudioProcessor,
        feature_extractor: FeatureExtractor,
        context_classifier: ContextClassifier,
        sensor_fusion: SensorFusion | None = None,
        resample_threshold: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.pf = particle_filter
        self.audio_proc = audio_processor
        self.feat_ext = feature_extractor
        self.classifier = context_classifier
        self.fusion = sensor_fusion or SensorFusion()
        self.resample_threshold = resample_threshold
        self.verbose = verbose

        self._step: int = 0
        # Store the base measurement noise so context scaling is non-cumulative
        self._base_measurement_noise_std: float = particle_filter.measurement_noise_std

    # ------------------------------------------------------------------
    # Main processing step
    # ------------------------------------------------------------------

    def step(
        self,
        delta_odom: tuple[float, float, float],
        audio_signal: np.ndarray,
        observations: list[dict] | None = None,
    ) -> SystemState:
        """Run one simultaneous localization + context-recognition cycle.

        Parameters
        ----------
        delta_odom : tuple ``(d_rot1, d_trans, d_rot2)``
            Odometry increment decoded from wheel encoders.
        audio_signal : numpy.ndarray, shape ``(N,)``
            Raw mono PCM audio captured during this time step.
        observations : list of dict, optional
            Range/bearing observations for the particle-filter measurement
            update.  Each dict must contain ``"range"``, ``"bearing"``, and
            ``"landmark"`` keys.

        Returns
        -------
        SystemState
            The fused state estimate for this time step.
        """
        # ---- Localization ------------------------------------------------
        pose = self.pf.step(
            delta_odom,
            observations=observations,
            resample_threshold=self.resample_threshold,
        )
        cov = self.pf.pose_covariance
        neff = self.pf.effective_sample_size

        # Optionally scale particle-filter noise by context from *previous* step
        if self.fusion.history:
            prev_context = self.fusion.history[-1].context
            scale = self.fusion.measurement_noise_scale(prev_context)
            # Apply scale against the *base* noise, not the already-scaled value,
            # so the scaling is idempotent and cannot grow unboundedly.
            self.pf.measurement_noise_std = max(0.05, scale * self._base_measurement_noise_std)

        # ---- Spatial context recognition ----------------------------------
        frames = self.audio_proc.process(audio_signal)
        power_spec = self.audio_proc.compute_power_spectrum(frames)
        features = self.feat_ext.extract(power_spec)
        context = self.classifier.predict(features)
        context_probs = self.classifier.predict_proba(features)

        # ---- Fusion -------------------------------------------------------
        state = self.fusion.fuse(
            pose=pose,
            pose_covariance=cov,
            context=context,
            context_probs=context_probs,
            step=self._step,
            effective_n_particles=neff,
        )

        if self.verbose:
            print(state)

        self._step += 1
        return state

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run(
        self,
        odometry_sequence: list[tuple[float, float, float]],
        audio_sequence: list[np.ndarray],
        observations_sequence: list[list[dict]] | None = None,
    ) -> list[SystemState]:
        """Process a full sequence of observations.

        Parameters
        ----------
        odometry_sequence : list of tuples
        audio_sequence : list of 1-D arrays
        observations_sequence : list of lists of dicts, optional

        Returns
        -------
        list of SystemState
        """
        if len(odometry_sequence) != len(audio_sequence):
            raise ValueError(
                "odometry_sequence and audio_sequence must have the same length."
            )
        if observations_sequence is None:
            observations_sequence = [None] * len(odometry_sequence)

        states = []
        for odom, audio, obs in zip(odometry_sequence, audio_sequence, observations_sequence):
            states.append(self.step(odom, audio, observations=obs))
        return states

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def current_pose(self) -> np.ndarray:
        """Most recent estimated pose ``[x, y, theta]``."""
        return self.pf.estimated_pose

    @property
    def current_context(self) -> str | None:
        """Most recent context label, or ``None`` if no steps taken."""
        if self.fusion.history:
            return self.fusion.history[-1].context
        return None

    @property
    def state_history(self) -> list[SystemState]:
        """Full list of :class:`SystemState` snapshots."""
        return self.fusion.history
