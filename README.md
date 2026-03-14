# Speech-Processing-and-Robotics
## Design of a Simultaneous Mobile Robot Localization and Spatial Context Recognition System

This repository implements a complete system that simultaneously performs:
1. **Mobile robot localization** – tracking the robot's 2-D pose (x, y, θ) using a Particle Filter (Monte Carlo Localization).
2. **Spatial context recognition** – classifying the acoustic environment (corridor, room, open space, staircase, outdoor) from short audio frames using MFCC features and a Gaussian Naïve Bayes classifier.

Both pipelines are fused in real time, with the current spatial context used to modulate the localization's measurement-noise model.

---

## Project Structure

```
.
├── localization/
│   ├── odometry.py          # Probabilistic differential-drive odometry model
│   └── particle_filter.py   # SIR Particle Filter (Monte Carlo Localization)
├── spatial_context/
│   ├── audio_processor.py   # Audio pre-processing (framing, windowing, power spectrum)
│   ├── feature_extraction.py# MFCC + room-acoustic feature extraction
│   └── context_classifier.py# Gaussian Naïve Bayes spatial context classifier
├── system/
│   ├── sensor_fusion.py     # State fusion, history management, trajectory smoothing
│   └── simultaneous_system.py # Top-level orchestrator (predict → recognise → fuse)
├── utils/
│   ├── math_utils.py        # Shared maths helpers (angle wrapping, Gaussian PDF, …)
│   └── visualization.py     # Matplotlib trajectory / particle / timeline plots
├── tests/
│   ├── test_localization.py
│   ├── test_spatial_context.py
│   └── test_system.py
├── main.py                  # End-to-end simulation demo
└── requirements.txt
```

---

## Algorithms

| Component | Algorithm |
|-----------|-----------|
| Localization | Particle Filter (SIR – Sequential Importance Resampling) |
| Motion model | Probabilistic odometry (Thrun, Burgard & Fox Ch. 5) |
| Resampling | Low-variance (systematic) resampling |
| Audio features | MFCCs (13 coefficients) + Δ + ΔΔ + 4 room-acoustic scalars |
| Context classifier | Gaussian Naïve Bayes (no external ML library required) |
| Fusion | Context-aware measurement-noise scaling + sliding state history |

---

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, NumPy, SciPy, Matplotlib.

---

## Running the Demo

```bash
python main.py
```

The script simulates a robot navigating through a corridor, a room, and an open space.
Results are saved to `results/`:

| File | Description |
|------|-------------|
| `trajectory.png` | Estimated vs. ground-truth path, colour-coded by context |
| `particles_final.png` | Particle cloud at the final step |
| `context_timeline.png` | Context classification over time with posterior probabilities |
| `summary.json` | Numerical summary (accuracy, final pose error, …) |

---

## Running the Tests

```bash
python -m pytest tests/ -v
```

49 unit tests cover the odometry model, particle filter, audio processor, feature extractor, context classifier, sensor fusion, and the simultaneous system integration.

---

## System Overview

```
         ┌─────────────────────────────────────────────────┐
         │            SimultaneousSystem.step()             │
         │                                                  │
  Wheel  │   ┌──────────────┐        ┌──────────────────┐  │
encoders │   │              │        │  AudioProcessor  │  │  Microphone
─────────┼──►│ ParticleFilter│        │  FeatureExtractor│◄─┼─────────────
         │   │   .step()    │        │  ContextClassifier│  │
         │   └──────┬───────┘        └────────┬─────────┘  │
         │          │  pose + cov             │  label + P  │
         │          └─────────┬───────────────┘             │
         │                    ▼                              │
         │             SensorFusion.fuse()                   │
         │                    │                              │
         │                    ▼                              │
         │              SystemState                          │
         └─────────────────────────────────────────────────┘
```
