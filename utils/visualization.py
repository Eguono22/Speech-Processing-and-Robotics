"""
Visualization utilities for the localization and context-recognition system.

All plots are generated with Matplotlib and saved to disk (non-interactive
mode) so the system can run headlessly on robots without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")   # headless backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False


class Visualizer:
    """Generate and save diagnostic plots for the simultaneous system.

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory where plot files are saved (default: current working directory).
    figure_size : tuple ``(width, height)``
        Figure size in inches.
    """

    def __init__(
        self,
        output_dir: str | Path = ".",
        figure_size: tuple[float, float] = (10.0, 8.0),
    ) -> None:
        if not _MPL_AVAILABLE:  # pragma: no cover
            raise ImportError("matplotlib is required for Visualizer.  Install it with: pip install matplotlib")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_size = figure_size

    # ------------------------------------------------------------------
    # Particle cloud + ground truth
    # ------------------------------------------------------------------

    def plot_particles(
        self,
        particles: np.ndarray,
        weights: np.ndarray,
        estimated_pose: np.ndarray,
        ground_truth: np.ndarray | None = None,
        landmarks: list[tuple[float, float]] | None = None,
        title: str = "Particle Filter – Robot Localization",
        filename: str = "particles.png",
    ) -> Path:
        """Plot the particle cloud and pose estimate.

        Parameters
        ----------
        particles : numpy.ndarray, shape ``(N, 3)``
        weights : numpy.ndarray, shape ``(N,)``
        estimated_pose : numpy.ndarray, shape ``(3,)``
        ground_truth : numpy.ndarray, shape ``(3,)``, optional
        landmarks : list of ``(x, y)`` tuples, optional
        title : str
        filename : str

        Returns
        -------
        pathlib.Path – path to the saved image
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Scale point sizes by weight
        sizes = np.clip(weights / weights.max() * 60, 1, 60)
        ax.scatter(particles[:, 0], particles[:, 1], s=sizes, c="steelblue",
                   alpha=0.5, label="Particles")

        # Estimated pose
        ax.scatter(*estimated_pose[:2], marker="*", s=200, c="darkorange",
                   zorder=5, label="Estimated pose")
        self._draw_arrow(ax, estimated_pose, color="darkorange")

        # Ground truth
        if ground_truth is not None:
            ax.scatter(*ground_truth[:2], marker="D", s=100, c="green",
                       zorder=5, label="Ground truth")
            self._draw_arrow(ax, ground_truth, color="green")

        # Landmarks
        if landmarks:
            lx, ly = zip(*landmarks)
            ax.scatter(lx, ly, marker="^", s=120, c="red", zorder=5, label="Landmarks")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

        out = self.output_dir / filename
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # Trajectory
    # ------------------------------------------------------------------

    def plot_trajectory(
        self,
        estimated_trajectory: Sequence[np.ndarray],
        ground_truth_trajectory: Sequence[np.ndarray] | None = None,
        landmarks: list[tuple[float, float]] | None = None,
        context_labels: list[str] | None = None,
        title: str = "Robot Trajectory",
        filename: str = "trajectory.png",
    ) -> Path:
        """Plot full estimated vs. ground-truth trajectory.

        Parameters
        ----------
        estimated_trajectory : sequence of pose arrays ``(3,)``
        ground_truth_trajectory : sequence of pose arrays, optional
        landmarks : list of ``(x, y)``, optional
        context_labels : list of str, optional
            Per-step context label.  When provided, trajectory segments are
            coloured by context.
        title, filename : str

        Returns
        -------
        pathlib.Path
        """
        fig, ax = plt.subplots(figsize=self.figure_size)

        est = np.array(estimated_trajectory)

        if context_labels:
            unique_labels = sorted(set(context_labels))
            cmap = plt.cm.get_cmap("tab10", len(unique_labels))
            colour_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}
            for i in range(len(est) - 1):
                colour = colour_map.get(context_labels[i], "gray")
                ax.plot(est[i:i + 2, 0], est[i:i + 2, 1], color=colour, linewidth=2)
            patches = [mpatches.Patch(color=colour_map[lbl], label=lbl) for lbl in unique_labels]
            ax.legend(handles=patches, loc="upper right")
        else:
            ax.plot(est[:, 0], est[:, 1], "b-", linewidth=2, label="Estimated")

        if ground_truth_trajectory is not None:
            gt = np.array(ground_truth_trajectory)
            ax.plot(gt[:, 0], gt[:, 1], "g--", linewidth=1.5, label="Ground truth")

        if landmarks:
            lx, ly = zip(*landmarks)
            ax.scatter(lx, ly, marker="^", s=120, c="red", zorder=5, label="Landmarks")

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        if not context_labels:
            ax.legend(loc="upper right")
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

        out = self.output_dir / filename
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # Context classification over time
    # ------------------------------------------------------------------

    def plot_context_timeline(
        self,
        context_labels: list[str],
        context_probs: list[dict[str, float]] | None = None,
        title: str = "Spatial Context Over Time",
        filename: str = "context_timeline.png",
    ) -> Path:
        """Plot context classification results across time steps.

        Parameters
        ----------
        context_labels : list of str
        context_probs : list of dicts ``{label: probability}``, optional
        title, filename : str

        Returns
        -------
        pathlib.Path
        """
        fig, axes = plt.subplots(
            1 + int(context_probs is not None),
            1,
            figsize=self.figure_size,
            squeeze=False,
        )

        unique_labels = sorted(set(context_labels))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        colour_map = {lbl: cmap(i) for i, lbl in enumerate(unique_labels)}

        # Top panel: colour-coded label bars
        ax0 = axes[0, 0]
        for t, lbl in enumerate(context_labels):
            ax0.barh(0, 1, left=t, color=colour_map[lbl], edgecolor="none")
        patches = [mpatches.Patch(color=colour_map[lbl], label=lbl) for lbl in unique_labels]
        ax0.legend(handles=patches, loc="upper right", fontsize=8)
        ax0.set_xlim(0, len(context_labels))
        ax0.set_yticks([])
        ax0.set_xlabel("Time step")
        ax0.set_title(title)

        # Bottom panel: probability traces
        if context_probs:
            ax1 = axes[1, 0]
            times = np.arange(len(context_probs))
            for lbl in unique_labels:
                probs = [cp.get(lbl, 0.0) for cp in context_probs]
                ax1.plot(times, probs, label=lbl, color=colour_map[lbl])
            ax1.set_xlabel("Time step")
            ax1.set_ylabel("Posterior probability")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.set_ylim(0, 1)
            ax1.grid(True, linestyle="--", alpha=0.4)

        fig.tight_layout()
        out = self.output_dir / filename
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_arrow(ax, pose: np.ndarray, length: float = 0.3, **kwargs) -> None:
        import math
        ax.annotate(
            "",
            xy=(pose[0] + length * math.cos(pose[2]),
                pose[1] + length * math.sin(pose[2])),
            xytext=(pose[0], pose[1]),
            arrowprops=dict(arrowstyle="->", color=kwargs.get("color", "black"), lw=1.5),
        )
