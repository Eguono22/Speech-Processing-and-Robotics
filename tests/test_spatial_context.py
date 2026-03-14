"""
Tests for the spatial context module
(AudioProcessor + FeatureExtractor + ContextClassifier).
"""

import numpy as np
import pytest

from spatial_context.audio_processor import AudioProcessor
from spatial_context.feature_extraction import FeatureExtractor
from spatial_context.context_classifier import ContextClassifier, CONTEXT_LABELS


# ---------------------------------------------------------------------------
# AudioProcessor
# ---------------------------------------------------------------------------

class TestAudioProcessor:
    def setup_method(self):
        self.proc = AudioProcessor(sample_rate=16_000, frame_length_ms=25, frame_step_ms=10)

    def _make_signal(self, duration=0.5):
        n = int(duration * 16_000)
        return np.random.default_rng(0).standard_normal(n)

    def test_process_output_shape(self):
        sig = self._make_signal()
        frames = self.proc.process(sig)
        assert frames.ndim == 2
        assert frames.shape[1] == self.proc.frame_length

    def test_process_non_empty(self):
        frames = self.proc.process(self._make_signal())
        assert frames.shape[0] > 0

    def test_process_rejects_2d_input(self):
        with pytest.raises(ValueError):
            self.proc.process(np.zeros((100, 2)))

    def test_power_spectrum_shape(self):
        sig = self._make_signal()
        frames = self.proc.process(sig)
        ps = self.proc.compute_power_spectrum(frames)
        assert ps.ndim == 2
        assert ps.shape[0] == frames.shape[0]

    def test_power_spectrum_non_negative(self):
        frames = self.proc.process(self._make_signal())
        ps = self.proc.compute_power_spectrum(frames)
        assert (ps >= 0).all()

    def test_no_high_pass(self):
        proc = AudioProcessor(high_pass_cutoff=0.0)
        frames = proc.process(self._make_signal())
        assert frames.shape[0] > 0

    def test_no_pre_emphasis(self):
        proc = AudioProcessor(pre_emphasis=0.0)
        frames = proc.process(self._make_signal())
        assert frames.shape[0] > 0


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def setup_method(self):
        self.proc = AudioProcessor(sample_rate=16_000)
        self.ext = FeatureExtractor(sample_rate=16_000, n_mfcc=13, n_fft=512)

    def _power_spectrum(self, seed=0):
        rng = np.random.default_rng(seed)
        sig = rng.standard_normal(int(0.5 * 16_000))
        frames = self.proc.process(sig)
        return self.proc.compute_power_spectrum(frames)

    def test_extract_returns_1d(self):
        feat = self.ext.extract(self._power_spectrum())
        assert feat.ndim == 1

    def test_extract_correct_length(self):
        feat = self.ext.extract(self._power_spectrum())
        assert len(feat) == self.ext.feature_dim

    def test_different_signals_differ(self):
        f1 = self.ext.extract(self._power_spectrum(0))
        f2 = self.ext.extract(self._power_spectrum(99))
        assert not np.allclose(f1, f2)

    def test_no_deltas(self):
        ext = FeatureExtractor(include_delta=False, include_delta2=False)
        feat = ext.extract(self._power_spectrum())
        assert len(feat) == ext.feature_dim


# ---------------------------------------------------------------------------
# ContextClassifier
# ---------------------------------------------------------------------------

class TestContextClassifier:
    def _make_data(self, n_per_class=20, n_features=30, seed=0):
        rng = np.random.default_rng(seed)
        X, y = [], []
        for i, lbl in enumerate(CONTEXT_LABELS):
            samples = rng.normal(loc=float(i), scale=0.3, size=(n_per_class, n_features))
            X.append(samples)
            y.extend([lbl] * n_per_class)
        return np.vstack(X), y

    def test_fit_and_predict(self):
        X, y = self._make_data()
        clf = ContextClassifier()
        clf.fit(X, y)
        # Predict on training data centroid of each class
        rng = np.random.default_rng(0)
        for i, lbl in enumerate(CONTEXT_LABELS):
            x = rng.normal(loc=float(i), scale=0.05, size=30)
            pred = clf.predict(x)
            assert pred == lbl

    def test_predict_proba_sums_to_one(self):
        X, y = self._make_data()
        clf = ContextClassifier().fit(X, y)
        probs = clf.predict_proba(X[0])
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_predict_proba_all_labels_present(self):
        X, y = self._make_data()
        clf = ContextClassifier().fit(X, y)
        probs = clf.predict_proba(X[0])
        for lbl in CONTEXT_LABELS:
            assert lbl in probs

    def test_fit_from_params_round_trip(self):
        X, y = self._make_data()
        clf = ContextClassifier().fit(X, y)
        params = clf.get_params()
        clf2 = ContextClassifier(labels=params["labels"])
        clf2.fit_from_params(params)
        assert clf.predict(X[0]) == clf2.predict(X[0])

    def test_untrained_raises(self):
        clf = ContextClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros(10))

    def test_all_labels_accessible(self):
        assert set(CONTEXT_LABELS) == {
            "corridor", "room", "open_space", "staircase", "outdoor"
        }
