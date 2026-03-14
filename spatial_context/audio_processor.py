"""
Audio pre-processing utilities.

Accepts raw PCM data (as a NumPy array) and produces short-time frame
matrices ready for feature extraction.  No heavy external dependencies
are required – only NumPy and SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt


class AudioProcessor:
    """Pre-process raw audio into overlapping frames.

    Parameters
    ----------
    sample_rate : int
        Sampling frequency in Hz (default: 16 000).
    frame_length_ms : float
        Frame length in milliseconds (default: 25 ms).
    frame_step_ms : float
        Step between successive frames in milliseconds (default: 10 ms).
    pre_emphasis : float
        Pre-emphasis filter coefficient (default: 0.97).
        Set to ``0`` to disable.
    high_pass_cutoff : float
        Cut-off frequency (Hz) for an optional high-pass filter applied
        before framing.  Set to ``0`` to disable (default: 80 Hz).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_length_ms: float = 25.0,
        frame_step_ms: float = 10.0,
        pre_emphasis: float = 0.97,
        high_pass_cutoff: float = 80.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_length = int(round(frame_length_ms * sample_rate / 1000))
        self.frame_step = int(round(frame_step_ms * sample_rate / 1000))
        self.pre_emphasis = pre_emphasis
        self.high_pass_cutoff = high_pass_cutoff

        if high_pass_cutoff > 0:
            self._sos = butter(
                4,
                high_pass_cutoff / (sample_rate / 2),
                btype="high",
                output="sos",
            )
        else:
            self._sos = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Pre-process *signal* and return a frame matrix.

        Parameters
        ----------
        signal : numpy.ndarray, shape ``(N,)``
            Mono PCM signal (any float or int dtype).

        Returns
        -------
        numpy.ndarray, shape ``(n_frames, frame_length)``
            Windowed frames (Hamming window applied).
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            raise ValueError("signal must be a 1-D array (mono).")

        # High-pass filter
        if self._sos is not None:
            signal = sosfilt(self._sos, signal)

        # Pre-emphasis
        if self.pre_emphasis > 0:
            signal = np.append(signal[0], signal[1:] - self.pre_emphasis * signal[:-1])

        # Pad so last frame is complete
        n_frames = 1 + max(0, (len(signal) - self.frame_length) // self.frame_step)
        pad_length = (n_frames - 1) * self.frame_step + self.frame_length
        signal = np.pad(signal, (0, max(0, pad_length - len(signal))))

        # Stack frames
        indices = (
            np.arange(self.frame_length)[None, :]
            + self.frame_step * np.arange(n_frames)[:, None]
        )
        frames = signal[indices]

        # Apply Hamming window
        frames *= np.hamming(self.frame_length)
        return frames

    def compute_power_spectrum(self, frames: np.ndarray, n_fft: int | None = None) -> np.ndarray:
        """Compute the one-sided power spectrum for each frame.

        Parameters
        ----------
        frames : numpy.ndarray, shape ``(n_frames, frame_length)``
        n_fft : int, optional
            FFT size (zero-padded); defaults to next power of 2 ≥ frame_length.

        Returns
        -------
        numpy.ndarray, shape ``(n_frames, n_fft // 2 + 1)``
        """
        if n_fft is None:
            n_fft = int(2 ** np.ceil(np.log2(frames.shape[1])))
        spec = np.fft.rfft(frames, n=n_fft)
        return (np.abs(spec) ** 2) / n_fft
