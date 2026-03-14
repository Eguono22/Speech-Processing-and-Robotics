"""
System integration package.

Combines the localization and spatial-context-recognition pipelines into a
single simultaneous processing loop and provides sensor-fusion utilities.
"""

from .sensor_fusion import SensorFusion
from .simultaneous_system import SimultaneousSystem

__all__ = ["SensorFusion", "SimultaneousSystem"]
