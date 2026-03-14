"""
Simultaneous Mobile Robot Localization and Spatial Context Recognition System
=============================================================================

Entry point / demonstration script.

Run with:
    python main.py

The script simulates a differential-drive robot navigating a simple 2-D
environment that includes a corridor, a room, and an open space.  At each
time step it:

* Advances the robot along a pre-defined waypoint path.
* Generates noisy odometry and synthetic range observations.
* Generates a synthetic audio signal whose acoustic properties vary with
  the current spatial context.
* Runs the full simultaneous localization + spatial context recognition
  pipeline.
* Saves trajectory and context-timeline plots to the ``results/`` directory.
"""

from __future__ import annotations

import math
import json
import numpy as np
from pathlib import Path

from localization.particle_filter import ParticleFilter
from localization.odometry import OdometryModel
from spatial_context.audio_processor import AudioProcessor
from spatial_context.feature_extraction import FeatureExtractor
from spatial_context.context_classifier import ContextClassifier, CONTEXT_LABELS
from system.sensor_fusion import SensorFusion
from system.simultaneous_system import SimultaneousSystem
from utils.visualization import Visualizer


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Hz
AUDIO_DURATION = 0.5          # seconds per step
N_PARTICLES = 300
MAP_BOUNDS = (0.0, 20.0, 0.0, 15.0)
RANDOM_SEED = 42

# Landmark positions (known a priori, used for range observations)
LANDMARKS: list[tuple[float, float]] = [
    (2.0, 2.0),
    (10.0, 2.0),
    (18.0, 2.0),
    (2.0, 13.0),
    (10.0, 7.5),
    (18.0, 13.0),
]

# Waypoints that define the robot's true path
WAYPOINTS: list[tuple[float, float]] = [
    (1.0, 1.0),
    (5.0, 1.0),
    (10.0, 1.0),    # corridor → room boundary
    (10.0, 5.0),
    (10.0, 10.0),   # room → open-space boundary
    (15.0, 10.0),
    (19.0, 10.0),
]

# Spatial context associated with each segment between waypoints
SEGMENT_CONTEXTS: list[str] = [
    "corridor",
    "corridor",
    "room",
    "room",
    "open_space",
    "open_space",
]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _interpolate_path(
    waypoints: list[tuple[float, float]],
    step_size: float = 0.3,
) -> list[tuple[float, float, float, str]]:
    """Densify waypoints into a list of ``(x, y, theta, context)`` poses."""
    path: list[tuple[float, float, float, str]] = []
    for seg_idx, (p0, p1) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        dist = math.hypot(dx, dy)
        n_steps = max(1, int(dist / step_size))
        theta = math.atan2(dy, dx)
        context = SEGMENT_CONTEXTS[seg_idx] if seg_idx < len(SEGMENT_CONTEXTS) else "room"
        for i in range(n_steps):
            t = i / n_steps
            x = p0[0] + t * dx
            y = p0[1] + t * dy
            path.append((x, y, theta, context))
    path.append((*waypoints[-1], 0.0, SEGMENT_CONTEXTS[-1]))
    return path


def _generate_range_observations(
    pose: tuple[float, float, float],
    landmarks: list[tuple[float, float]],
    max_range: float = 8.0,
    noise_std: float = 0.2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Simulate noisy range + bearing observations to nearby landmarks."""
    if rng is None:
        rng = np.random.default_rng()
    observations = []
    for lx, ly in landmarks:
        dx, dy = lx - pose[0], ly - pose[1]
        true_range = math.hypot(dx, dy)
        if true_range > max_range:
            continue
        true_bearing = math.atan2(dy, dx) - pose[2]
        obs = {
            "range": true_range + rng.normal(0, noise_std),
            "bearing": true_bearing + rng.normal(0, 0.05),
            "landmark": (lx, ly),
        }
        observations.append(obs)
    return observations


def _generate_audio_signal(
    context: str,
    duration: float,
    sample_rate: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Synthesize a contextually distinct audio signal.

    Different contexts are distinguished by their spectral envelope and
    energy distribution, mimicking real room-acoustic differences:
    * ``corridor`` – strong low-frequency resonances, high reverberation.
    * ``room``      – balanced spectrum, moderate reverberation.
    * ``open_space``– flat, low-energy background; no resonances.
    * ``staircase`` – flutter echo, pronounced mid-frequency peaks.
    * ``outdoor``   – wind noise (band-limited); low spectral flatness.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)

    # Base noise
    signal = rng.standard_normal(n)

    context_profiles: dict[str, dict] = {
        "corridor": {"freqs": [120, 240, 360], "amps": [0.8, 0.4, 0.2], "noise_scale": 0.3},
        "room":     {"freqs": [300, 600, 900], "amps": [0.5, 0.3, 0.15], "noise_scale": 0.5},
        "open_space": {"freqs": [80], "amps": [0.1], "noise_scale": 0.9},
        "staircase":  {"freqs": [500, 1000, 1500], "amps": [0.6, 0.35, 0.2], "noise_scale": 0.35},
        "outdoor":    {"freqs": [200, 400], "amps": [0.3, 0.15], "noise_scale": 0.7},
    }

    profile = context_profiles.get(context, context_profiles["room"])
    for freq, amp in zip(profile["freqs"], profile["amps"]):
        signal += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    signal *= profile["noise_scale"]
    signal /= (np.abs(signal).max() + 1e-10)  # Normalise to [-1, 1]
    return signal


# ---------------------------------------------------------------------------
# Classifier bootstrap (synthetic training)
# ---------------------------------------------------------------------------

def _train_classifier(
    feature_extractor: FeatureExtractor,
    audio_proc: AudioProcessor,
    n_samples_per_class: int = 40,
    rng: np.random.Generator | None = None,
) -> ContextClassifier:
    """Train a GNB classifier on synthetic audio samples."""
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    X_train, y_train = [], []
    for label in CONTEXT_LABELS:
        for _ in range(n_samples_per_class):
            sig = _generate_audio_signal(label, AUDIO_DURATION, SAMPLE_RATE, rng)
            frames = audio_proc.process(sig)
            power = audio_proc.compute_power_spectrum(frames)
            feat = feature_extractor.extract(power)
            X_train.append(feat)
            y_train.append(label)

    classifier = ContextClassifier()
    classifier.fit(np.array(X_train), y_train)
    return classifier


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print(" Simultaneous Mobile Robot Localization &")
    print(" Spatial Context Recognition System")
    print("=" * 65)

    # ---- Build pipeline components -----------------------------------------
    odometry_model = OdometryModel(alpha1=0.02, alpha2=0.02, alpha3=0.02, alpha4=0.02)
    pf = ParticleFilter(
        n_particles=N_PARTICLES,
        map_bounds=MAP_BOUNDS,
        odometry_model=odometry_model,
        measurement_noise_std=0.4,
        rng=np.random.default_rng(RANDOM_SEED),
    )

    audio_proc = AudioProcessor(sample_rate=SAMPLE_RATE, frame_length_ms=25, frame_step_ms=10)
    feat_ext = FeatureExtractor(sample_rate=SAMPLE_RATE, n_mfcc=13, n_fft=512)
    sensor_fusion = SensorFusion()

    # ---- Train context classifier ------------------------------------------
    print("\n[1/4] Training spatial context classifier …")
    classifier = _train_classifier(feat_ext, audio_proc, rng=rng)
    print(f"      Classifier trained on {len(CONTEXT_LABELS)} classes: {CONTEXT_LABELS}")

    # ---- Build simultaneous system -----------------------------------------
    system = SimultaneousSystem(
        particle_filter=pf,
        audio_processor=audio_proc,
        feature_extractor=feat_ext,
        context_classifier=classifier,
        sensor_fusion=sensor_fusion,
        verbose=True,
    )

    # ---- Generate path -----------------------------------------------------
    path = _interpolate_path(WAYPOINTS, step_size=0.3)
    print(f"\n[2/4] Simulating {len(path)} steps along a {len(WAYPOINTS)-1}-segment path …\n")

    # Initialise PF near true start pose
    system.pf.reset(initial_pose=(path[0][0], path[0][1], path[0][2]), spread=0.3)

    # ---- Run simulation loop -----------------------------------------------
    ground_truth: list[np.ndarray] = []
    prev_pose = path[0][:3]

    for i, (x, y, theta, context) in enumerate(path):
        true_pose = (x, y, theta)
        delta_odom = OdometryModel.compute_odometry(prev_pose, true_pose)
        audio = _generate_audio_signal(context, AUDIO_DURATION, SAMPLE_RATE, rng)
        obs = _generate_range_observations(true_pose, LANDMARKS, rng=rng)
        system.step(delta_odom, audio, observations=obs)
        ground_truth.append(np.array(true_pose))
        prev_pose = true_pose

    print(f"\n[3/4] Completed {len(path)} steps.")

    # ---- Save results -------------------------------------------------------
    print("\n[4/4] Saving results …")
    vis = Visualizer(output_dir=results_dir)

    traj_path = vis.plot_trajectory(
        estimated_trajectory=system.fusion.pose_trajectory(),
        ground_truth_trajectory=ground_truth,
        landmarks=LANDMARKS,
        context_labels=system.fusion.context_sequence(),
        title="Simultaneous Localization & Context Recognition – Trajectory",
        filename="trajectory.png",
    )
    print(f"      Trajectory plot → {traj_path}")

    ctx_path = vis.plot_context_timeline(
        context_labels=system.fusion.context_sequence(),
        context_probs=system.fusion.context_probability_sequence(),
        filename="context_timeline.png",
    )
    print(f"      Context timeline → {ctx_path}")

    # Final particle snapshot
    particle_path = vis.plot_particles(
        particles=system.pf.particles,
        weights=system.pf.weights,
        estimated_pose=system.pf.estimated_pose,
        ground_truth=ground_truth[-1],
        landmarks=LANDMARKS,
        title="Particle Filter – Final Step",
        filename="particles_final.png",
    )
    print(f"      Particle cloud  → {particle_path}")

    # Save summary JSON
    summary = {
        "n_steps": len(path),
        "final_estimated_pose": system.pf.estimated_pose.tolist(),
        "final_ground_truth": ground_truth[-1].tolist(),
        "context_sequence": system.fusion.context_sequence(),
        "context_accuracy": float(
            sum(
                1
                for s, gt in zip(system.fusion.context_sequence(), [p[3] for p in path])
                if s == gt
            )
            / len(path)
        ),
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Summary JSON    → {summary_path}")

    print("\n  Context classification accuracy : "
          f"{summary['context_accuracy'] * 100:.1f} %")
    est = system.pf.estimated_pose
    gt = ground_truth[-1]
    pos_error = math.hypot(est[0] - gt[0], est[1] - gt[1])
    print(f"  Final position error            : {pos_error:.3f} m")
    print("\nDone.")


if __name__ == "__main__":
    main()
