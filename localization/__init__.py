"""
Localization package.

Provides particle-filter-based Monte Carlo Localization (MCL) for a mobile
robot operating in a 2-D environment.
"""

from .odometry import OdometryModel
from .particle_filter import ParticleFilter

__all__ = ["OdometryModel", "ParticleFilter"]
