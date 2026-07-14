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


SEAT_ORDER = ("A-1", "A-2", "A-3", "A-4")
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
TABLE_A = {
    "left": -3.3,
    "bottom": 4.6,
    "width": 1.5,
    "height": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize A-seat table regions with alpha shapes from representative points."
    )
    parser.add_argument(
        "--points",
        default="output/a_representative_density_first_stop_filtered_direct_slowdown_only/seat_area_representative_points.csv",
    )
    parser.add_argument("--output-dir", default="output/a_alpha_shape_first_stop_filtered_direct_slowdown_only")
    parser.add_argument("--alpha-values", default="0.9,1.5")
    parser.add_argument("--min-points", type=int, default=4)
    args = parser.parse_args()

    points = pd.read_csv(args.points)
    points = points[points["seat"].isin(SEAT_ORDER)].copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    limits = _limits(points)
    summary_rows = []
    for alpha in _parse_alpha_values(args.alpha_values):
        alpha_label = _alpha_label(alpha)
        summary_rows.extend(_summarize(points, alpha, args.min_points))
        _plot_grid(
            points,
            alpha,
            args.min_points,
            output_dir / f"A_alpha_shape_by_seat_2x2_alpha{alpha_label}.png",
            limits,
        )
        _plot_all(
            points,
            alpha,
            args.min_points,
            output_dir / f"A_alpha_shape_all_seats_alpha{alpha_label}.png",
            limits,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "alpha_shape_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary: {summary_path}")
    for alpha in _parse_alpha_values(args.alpha_values):
        alpha_label = _alpha_label(alpha)
        print(f"2x2 alpha={alpha}: {output_dir / f'A_alpha_shape_by_seat_2x2_alpha{alpha_label}.png'}")
        print(f"All alpha={alpha}: {output_dir / f'A_alpha_shape_all_seats_alpha{alpha_label}.png'}")


def _parse_alpha_values(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _alpha_label(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def _summarize(points: pd.DataFrame, alpha: float, min_points: int) -> list[dict[str, float | int | str]]:
    rows = []
    for seat in SEAT_ORDER:
        group = points[points["seat"] == seat]
        coords = group[["x", "y"]].to_numpy(dtype=float)
        shape = alpha_shape(coords, alpha=alpha, min_points=min_points)
        rows.append(
            {
                "seat": seat,
                "alpha": float(alpha),
                "point_count": int(len(group)),
                "triangle_count": int(len(shape.triangles)),
                "alpha_area_m2": float(shape.area_m2),
                "centroid_x": float(coords[:, 0].mean()) if len(coords) else np.nan,
                "centroid_y": float(coords[:, 1].mean()) if len(coords) else np.nan,
            }
        )
    return rows


def _plot_grid(
    points: pd.DataFrame,
    alpha: float,
    min_points: int,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, seat in zip(axes.ravel(), SEAT_ORDER):
        group = points[points["seat"] == seat]
        _draw_alpha_shape(ax, group, seat, alpha, min_points)
        ax.set_title(f"{seat}  alpha={alpha:g}  n={len(group)}")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue", markersize=7, label="representatives"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="true seat"),
        plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="black", markeredgecolor="white", markersize=10, label="mean representative"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("representative x [m]")
    fig.supylabel("representative y [m]")
    fig.suptitle("Alpha-shape table-region estimate by A-seat", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_all(
    points: pd.DataFrame,
    alpha: float,
    min_points: int,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for seat in SEAT_ORDER:
        group = points[points["seat"] == seat]
        _draw_alpha_shape(ax, group, seat, alpha, min_points, label=True)
    ax.set_title(f"All A-seat alpha-shape regions  alpha={alpha:g}")
    ax.set_xlabel("representative x [m]")
    ax.set_ylabel("representative y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_alpha_shape(
    ax: plt.Axes,
    group: pd.DataFrame,
    seat: str,
    alpha: float,
    min_points: int,
    *,
    label: bool = False,
) -> None:
    color = COLORS[seat]
    coords = group[["x", "y"]].to_numpy(dtype=float)
    shape = alpha_shape(coords, alpha=alpha, min_points=min_points)
    if len(shape.triangles):
        collection = PolyCollection(
            shape.triangles,
            facecolors=color,
            edgecolors=color,
            linewidths=1.2,
            alpha=0.22,
            zorder=3,
            label=f"{seat} alpha shape" if label else None,
        )
        ax.add_collection(collection)
    _draw_table_a(ax)
    ax.scatter(
        group["x"],
        group["y"],
        s=28,
        color=color,
        alpha=0.70,
        edgecolors="white",
        linewidths=0.35,
        label=f"{seat} representatives" if label else None,
        zorder=6,
    )
    target_x, target_y = TARGETS[seat]
    mean_x = float(group["x"].mean())
    mean_y = float(group["y"].mean())
    ax.scatter([target_x], [target_y], marker="*", s=230, color="gold", edgecolor="black", zorder=7)
    ax.scatter([mean_x], [mean_y], marker="X", s=120, color="black", edgecolor="white", zorder=8)
    ax.annotate(seat, (target_x, target_y), xytext=(5, 5), textcoords="offset points", fontsize=9, zorder=9)


def _draw_table_a(ax: plt.Axes) -> None:
    rectangle = plt.Rectangle(
        (TABLE_A["left"], TABLE_A["bottom"]),
        TABLE_A["width"],
        TABLE_A["height"],
        fill=False,
        edgecolor="black",
        linewidth=3.0,
        zorder=4,
    )
    ax.add_patch(rectangle)


def _limits(points: pd.DataFrame) -> tuple[float, float, float, float]:
    xs = points["x"].to_list()
    ys = points["y"].to_list()
    for x, y in TARGETS.values():
        xs.append(x)
        ys.append(y)
    xs.extend([TABLE_A["left"], TABLE_A["left"] + TABLE_A["width"]])
    ys.extend([TABLE_A["bottom"], TABLE_A["bottom"] + TABLE_A["height"]])
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.55, (x_max - x_min) * 0.08)
    pad_y = max(0.55, (y_max - y_min) * 0.08)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


if __name__ == "__main__":
    main()
