from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.table_region import alpha_shape


TABLE_A = {
    "left": -3.3,
    "bottom": 4.6,
    "width": 1.5,
    "height": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a single alpha-shape region from all A-table-area segment points."
    )
    parser.add_argument(
        "--points",
        default="output/a_segments_first_stop_filtered_direct_slowdown_only/segment_points.csv",
    )
    parser.add_argument("--output-dir", default="output/a_table_segment_alpha_shape_first_stop_filtered")
    parser.add_argument("--alpha-values", default="0.9,1.5")
    parser.add_argument("--min-points", type=int, default=4)
    args = parser.parse_args()

    points = pd.read_csv(args.points)
    points = points.dropna(subset=["x", "y"]).copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limits = _limits(points)
    summary_rows = []
    for alpha in _parse_alpha_values(args.alpha_values):
        shape = alpha_shape(points[["x", "y"]].to_numpy(dtype=float), alpha=alpha, min_points=args.min_points)
        summary_rows.append(
            {
                "alpha": float(alpha),
                "point_count": int(len(points)),
                "triangle_count": int(len(shape.triangles)),
                "alpha_area_m2": float(shape.area_m2),
                "point_mean_x": float(points["x"].mean()),
                "point_mean_y": float(points["y"].mean()),
            }
        )
        output_path = output_dir / f"A_table_segment_alpha_shape_alpha{_alpha_label(alpha)}.png"
        _plot(points, shape, alpha, output_path, limits)
        print(f"Alpha-shape alpha={alpha}: {output_path}")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "table_segment_alpha_shape_summary.csv"
    summary.to_csv(summary_path, index=False)
    if len(summary_rows) >= 4:
        combined_path = output_dir / "A_table_segment_alpha_shape_alpha_2x2.png"
        _plot_alpha_grid(points, summary_rows[:4], combined_path, limits, args.min_points)
        print(f"Alpha-shape 2x2: {combined_path}")
    print(f"Summary: {summary_path}")


def _parse_alpha_values(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def _plot(
    points: pd.DataFrame,
    shape,
    alpha: float,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    if len(shape.triangles):
        collection = PolyCollection(
            shape.triangles,
            facecolors="tab:blue",
            edgecolors="tab:blue",
            linewidths=1.3,
            alpha=0.22,
            zorder=3,
            label="alpha-shape region",
        )
        ax.add_collection(collection)
    _draw_table_a(ax)
    ax.scatter(
        points["x"],
        points["y"],
        s=22,
        color="0.25",
        alpha=0.60,
        edgecolors="white",
        linewidths=0.25,
        label="table-area segment points",
        zorder=6,
    )
    ax.set_title(f"A-table-area segment alpha-shape  alpha={alpha:g}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_alpha_grid(
    points: pd.DataFrame,
    summary_rows: list[dict[str, float]],
    output_path: Path,
    limits: tuple[float, float, float, float],
    min_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, row in zip(axes.ravel(), summary_rows):
        alpha = float(row["alpha"])
        shape = alpha_shape(points[["x", "y"]].to_numpy(dtype=float), alpha=alpha, min_points=min_points)
        if len(shape.triangles):
            collection = PolyCollection(
                shape.triangles,
                facecolors="tab:blue",
                edgecolors="tab:blue",
                linewidths=1.0,
                alpha=0.24,
                zorder=3,
            )
            ax.add_collection(collection)
        _draw_table_a(ax, label=None)
        ax.scatter(
            points["x"],
            points["y"],
            s=16,
            color="0.25",
            alpha=0.55,
            edgecolors="white",
            linewidths=0.2,
            zorder=6,
        )
        ax.set_title(f"alpha={alpha:g}  area={float(row['alpha_area_m2']):.2f} m2")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([0], [0], color="tab:blue", linewidth=8, alpha=0.35, label="alpha-shape"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="0.25", markersize=6, label="segment points"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    fig.suptitle("A-table-area segment alpha-shape comparison", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_table_a(ax: plt.Axes, label: str | None = "table A") -> None:
    rectangle = plt.Rectangle(
        (TABLE_A["left"], TABLE_A["bottom"]),
        TABLE_A["width"],
        TABLE_A["height"],
        fill=False,
        edgecolor="black",
        linewidth=3.0,
        zorder=4,
        label=label,
    )
    ax.add_patch(rectangle)


def _limits(points: pd.DataFrame) -> tuple[float, float, float, float]:
    xs = points["x"].to_list()
    ys = points["y"].to_list()
    xs.extend([TABLE_A["left"], TABLE_A["left"] + TABLE_A["width"]])
    ys.extend([TABLE_A["bottom"], TABLE_A["bottom"] + TABLE_A["height"]])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.55, (x_max - x_min) * 0.08)
    pad_y = max(0.55, (y_max - y_min) * 0.08)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


if __name__ == "__main__":
    main()
