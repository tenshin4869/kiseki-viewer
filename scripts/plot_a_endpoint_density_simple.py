from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


TARGETS = {
    "A-1": (-2.0, 4.6),
    "A-2": (-2.7, 4.6),
    "A-3": (-2.0, 5.5),
    "A-4": (-2.7, 5.5),
}

COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple endpoint density figures.")
    parser.add_argument("--manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_endpoint_density_simple")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = pd.read_csv(args.manifest)
    endpoints = endpoints[endpoints["trajectory_name"].isin(TARGETS)].copy()

    heatmap_path = output_dir / "endpoint_density_hist2d_all.png"
    kde_path = output_dir / "endpoint_density_kde_by_seat.png"
    _plot_hist2d_all(endpoints, heatmap_path)
    _plot_kde_by_seat(endpoints, kde_path)

    print(f"2D histogram density: {heatmap_path}")
    print(f"KDE contour density: {kde_path}")


def _plot_hist2d_all(endpoints: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    hist = ax.hist2d(
        endpoints["endpoint_x"],
        endpoints["endpoint_y"],
        bins=28,
        cmap="YlOrRd",
    )
    fig.colorbar(hist[3], ax=ax, label="endpoint count")
    for label, group in endpoints.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=14,
            alpha=0.45,
            color=COLORS[label],
            label=label,
            edgecolors="none",
        )
    _draw_targets(ax)
    ax.set_title("A-seat endpoint density: 2D histogram")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_kde_by_seat(endpoints: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    x_min, x_max = endpoints["endpoint_x"].min() - 0.7, endpoints["endpoint_x"].max() + 0.7
    y_min, y_max = endpoints["endpoint_y"].min() - 0.7, endpoints["endpoint_y"].max() + 0.7
    xx, yy = np.mgrid[x_min:x_max:160j, y_min:y_max:160j]
    grid = np.vstack([xx.ravel(), yy.ravel()])

    for label, group in endpoints.groupby("trajectory_name", sort=True):
        points = group[["endpoint_x", "endpoint_y"]].to_numpy(dtype=float).T
        if points.shape[1] < 3:
            continue
        kde = gaussian_kde(points)
        density = kde(grid).reshape(xx.shape)
        levels = _positive_levels(density)
        if len(levels):
            ax.contour(
                xx,
                yy,
                density,
                levels=levels,
                colors=COLORS[label],
                linewidths=1.4,
                alpha=0.85,
            )
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=13,
            alpha=0.32,
            color=COLORS[label],
            label=label,
            edgecolors="none",
        )

    _draw_targets(ax)
    ax.set_title("A-seat endpoint density: KDE contours")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _positive_levels(density: np.ndarray) -> np.ndarray:
    positive = density[density > 0]
    if positive.size == 0:
        return np.array([])
    return np.percentile(positive, [60, 75, 90])


def _draw_targets(ax: plt.Axes) -> None:
    for label, (target_x, target_y) in TARGETS.items():
        ax.scatter(
            [target_x],
            [target_y],
            marker="*",
            s=190,
            color=COLORS[label],
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(label, (target_x, target_y), xytext=(5, 5), textcoords="offset points")


if __name__ == "__main__":
    main()
