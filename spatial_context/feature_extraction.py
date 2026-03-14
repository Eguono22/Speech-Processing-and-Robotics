"""
Acoustic feature extraction for spatial context recognition.

Extracts Mel-Frequency Cepstral Coefficients (MFCCs) and several
complementary room-acoustic features from a power spectrum matrix
produced by :class:`~spatial_context.audio_processor.AudioProcessor`.

All computations use NumPy / SciPy only – no external speech-toolkit
dependency is required.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct


# ---------------------------------------------------------------------------
# Mel filterbank helpers
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Build a Mel filterbank matrix.

    Returns
    -------
    numpy.ndarray, shape ``(n_filters, n_fft // 2 + 1)``
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_bins))
    for m in range(1, n_filters + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        for k in range(f_left, f_center + 1):
            if f_center != f_left:
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right + 1):
            if f_right != f_center:
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filterbank


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract MFCCs and room-acoustic features from power-spectrum frames.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz.
    n_mfcc : int
        Number of MFCC coefficients to retain (default: 13).
    n_mels : int
        Number of Mel filterbank channels (default: 26).
    n_fft : int
        FFT size used to produce the power spectra (default: 512).
    include_delta : bool
        Append first-order delta (velocity) coefficients (default: True).
    include_delta2 : bool
        Append second-order delta (acceleration) coefficients (default: True).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        n_mfcc: int = 13,
        n_mels: int = 26,
        n_fft: int = 512,
        include_delta: bool = True,
        include_delta2: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.include_delta = include_delta
        self.include_delta2 = include_delta2

        self._filterbank = _mel_filterbank(n_mels, n_fft, sample_rate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Compute the feature vector for a sequence of frames.

        Parameters
        ----------
        power_spectrum : numpy.ndarray, shape ``(n_frames, n_fft // 2 + 1)``

        Returns
        -------
        numpy.ndarray, shape ``(feature_dim,)``
            Concatenated mean and standard deviation of MFCCs (+ deltas)
            and supplementary room-acoustic statistics, suitable for a
            downstream classifier.
        """
        mfcc = self._compute_mfcc(power_spectrum)          # (n_frames, n_mfcc)
        features = [mfcc]

        if self.include_delta:
            features.append(self._delta(mfcc))
        if self.include_delta2:
            features.append(self._delta(self._delta(mfcc)))

        feat_matrix = np.concatenate(features, axis=1)     # (n_frames, D)

        # Summarise over time: mean + std
        summary = np.concatenate([feat_matrix.mean(axis=0), feat_matrix.std(axis=0)])

        # Append room-acoustic features
        room_feats = self._room_acoustic_features(power_spectrum)
        return np.concatenate([summary, room_feats])

    @property
    def feature_dim(self) -> int:
        """Total length of the feature vector returned by :meth:`extract`."""
        n_coeff = self.n_mfcc * (1 + int(self.include_delta) + int(self.include_delta2))
        return 2 * n_coeff + 4   # mean+std + 4 room features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_mfcc(self, power_spectrum: np.ndarray) -> np.ndarray:
        # Apply Mel filterbank
        mel_energy = power_spectrum @ self._filterbank.T           # (n_frames, n_mels)
        mel_energy = np.maximum(mel_energy, 1e-10)
        log_mel = np.log(mel_energy)

        # DCT-II to get cepstral coefficients
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :self.n_mfcc]
        return mfcc

    @staticmethod
    def _delta(coeff: np.ndarray, width: int = 2) -> np.ndarray:
        """Compute first-order delta features using a regression window."""
        n = coeff.shape[0]
        delta = np.zeros_like(coeff)
        denom = 2 * sum(k ** 2 for k in range(1, width + 1))
        for t in range(n):
            for k in range(1, width + 1):
                t_fwd = min(t + k, n - 1)
                t_bwd = max(t - k, 0)
                delta[t] += k * (coeff[t_fwd] - coeff[t_bwd])
        delta /= (denom if denom != 0 else 1)
        return delta

    def _room_acoustic_features(self, power_spectrum: np.ndarray) -> np.ndarray:
        """Four scalar room-acoustic features.

        Returns ``[spectral_centroid, spectral_flatness,
                   spectral_rolloff, energy_ratio_low_high]``
        """
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        mean_power = power_spectrum.mean(axis=0)
        total_power = mean_power.sum() + 1e-10

        # Spectral centroid
        centroid = float(np.dot(freqs, mean_power) / total_power)

        # Spectral flatness (geometric / arithmetic mean)
        log_mean = np.exp(np.mean(np.log(mean_power + 1e-10)))
        arith_mean = total_power / len(mean_power)
        flatness = float(log_mean / (arith_mean + 1e-10))

        # Spectral roll-off (frequency below which 85 % of energy lies)
        cumsum = np.cumsum(mean_power)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * total_power)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Low-vs-high energy ratio (split at 1 kHz)
        split = np.searchsorted(freqs, 1000.0)
        low_energy = mean_power[:split].sum()
        high_energy = mean_power[split:].sum() + 1e-10
        energy_ratio = float(low_energy / high_energy)

        return np.array([centroid, flatness, rolloff, energy_ratio])
