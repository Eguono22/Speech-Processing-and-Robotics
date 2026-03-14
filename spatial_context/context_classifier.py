"""
Spatial context classifier.

Uses a Gaussian Naïve Bayes (GNB) model to classify acoustic feature
vectors into discrete spatial context categories such as *corridor*,
*room*, *open space*, etc.  The model can be trained incrementally or
loaded from a pre-saved parameter dictionary so that no external
ML library is required.

Supported context labels
------------------------
``"corridor"``  – long, reverberant, narrow spaces
``"room"``      – medium-sized enclosed rooms
``"open_space"``– large, low-reverberation environments
``"staircase"`` – stairwells (distinctive echo pattern)
``"outdoor"``   – outdoor / unenclosed areas
"""

from __future__ import annotations

from typing import Sequence
import numpy as np


# ---------------------------------------------------------------------------
# Context label definitions
# ---------------------------------------------------------------------------

CONTEXT_LABELS: list[str] = [
    "corridor",
    "room",
    "open_space",
    "staircase",
    "outdoor",
]


# ---------------------------------------------------------------------------
# Gaussian Naïve Bayes classifier
# ---------------------------------------------------------------------------

class ContextClassifier:
    """Gaussian Naïve Bayes classifier for spatial context recognition.

    Parameters
    ----------
    labels : list of str, optional
        Ordered list of class labels.  Defaults to :data:`CONTEXT_LABELS`.
    var_smoothing : float
        Additive smoothing applied to per-feature variances to avoid
        zero-variance issues (default: ``1e-9``).
    """

    def __init__(
        self,
        labels: list[str] | None = None,
        var_smoothing: float = 1e-9,
    ) -> None:
        self.labels = labels or CONTEXT_LABELS[:]
        self.var_smoothing = var_smoothing
        self._n_classes = len(self.labels)
        self._label_to_idx: dict[str, int] = {lbl: i for i, lbl in enumerate(self.labels)}

        # Parameters – set during training
        self._means: np.ndarray | None = None       # (n_classes, n_features)
        self._vars: np.ndarray | None = None        # (n_classes, n_features)
        self._log_priors: np.ndarray | None = None  # (n_classes,)
        self._is_trained: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "ContextClassifier":
        """Fit the model from labelled feature vectors.

        Parameters
        ----------
        X : numpy.ndarray, shape ``(n_samples, n_features)``
        y : sequence of str
            Class labels for each sample.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y_arr = np.asarray([self._label_to_idx[lbl] for lbl in y], dtype=int)
        n_features = X.shape[1]

        self._means = np.zeros((self._n_classes, n_features))
        self._vars = np.zeros((self._n_classes, n_features))
        self._log_priors = np.zeros(self._n_classes)

        for c in range(self._n_classes):
            mask = y_arr == c
            X_c = X[mask]
            if X_c.shape[0] == 0:
                continue
            self._means[c] = X_c.mean(axis=0)
            self._vars[c] = X_c.var(axis=0)
            self._log_priors[c] = np.log(mask.sum() / len(y_arr))

        self._is_trained = True
        return self

    def fit_from_params(self, params: dict) -> "ContextClassifier":
        """Load pre-computed parameters (means, variances, log-priors).

        Parameters
        ----------
        params : dict with keys ``"means"``, ``"vars"``, ``"log_priors"``

        Returns
        -------
        self
        """
        self._means = np.asarray(params["means"], dtype=float)
        self._vars = np.asarray(params["vars"], dtype=float)
        self._log_priors = np.asarray(params["log_priors"], dtype=float)
        self._is_trained = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x: np.ndarray) -> str:
        """Predict the spatial context label for feature vector *x*.

        Parameters
        ----------
        x : numpy.ndarray, shape ``(n_features,)``

        Returns
        -------
        str  – the predicted context label
        """
        log_posteriors = self._log_posteriors(np.asarray(x, dtype=float))
        return self.labels[int(np.argmax(log_posteriors))]

    def predict_proba(self, x: np.ndarray) -> dict[str, float]:
        """Return a probability distribution over context labels.

        Parameters
        ----------
        x : numpy.ndarray, shape ``(n_features,)``

        Returns
        -------
        dict mapping each label to its posterior probability
        """
        log_post = self._log_posteriors(np.asarray(x, dtype=float))
        # Normalise in log-space for numerical stability
        log_post -= log_post.max()
        probs = np.exp(log_post)
        probs /= probs.sum()
        return {lbl: float(p) for lbl, p in zip(self.labels, probs)}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """Return a dict of model parameters (suitable for serialisation)."""
        self._check_trained()
        return {
            "means": self._means.tolist(),
            "vars": self._vars.tolist(),
            "log_priors": self._log_priors.tolist(),
            "labels": self.labels,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("The classifier has not been trained yet.  Call fit() first.")

    def _log_posteriors(self, x: np.ndarray) -> np.ndarray:
        self._check_trained()
        log_likelihoods = np.zeros(self._n_classes)
        smoothed_vars = self._vars + self.var_smoothing
        for c in range(self._n_classes):
            diff = x - self._means[c]
            log_likelihoods[c] = -0.5 * np.sum(
                np.log(2 * np.pi * smoothed_vars[c]) + diff ** 2 / smoothed_vars[c]
            )
        return log_likelihoods + self._log_priors
