"""
Spatial context recognition package.

Provides acoustic-feature extraction and a lightweight classifier that
identifies the type of space the robot currently occupies (e.g. corridor,
room, open space) from short audio frames.
"""

from .audio_processor import AudioProcessor
from .feature_extraction import FeatureExtractor
from .context_classifier import ContextClassifier

__all__ = ["AudioProcessor", "FeatureExtractor", "ContextClassifier"]
