from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.table_region import _dbscan


TABLE_A = {
    "left": -3.3,
    "bottom": 4.6,
    "width": 1.5,
    "height": 0.9,
}
TRUE_CENTER = (-2.55, 5.05)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DBSCAN-estimated points from first-stop endpoints and seat-area segment representatives."
    )
    parser.add_argument("--manifest", default="output/a_trajectories_first_stop_filtered/manifest.csv")
    parser.add_argument("--segments", default="output/a_segments_first_stop_filtered_direct_slowdown_only/segments.csv")
    parser.add_argument("--output-dir", default="output/a_dbscan_estimate_comparison_first_stop_filtered")
    parser.add_argument("--eps-m", type=float, default=1.2)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    endpoint_points = manifest.rename(
        columns={"endpoint_x": "x", "endpoint_y": "y"}
    )[["trial_id", "trajectory_name", "x", "y", "trajectory_csv"]].copy()
    endpoint_points["method"] = "endpoint"

    segments = pd.read_csv(args.segments)
    representative_points = segments[segments["classification"] == "seat_area"].rename(
        columns={"destination_label": "trajectory_name", "representative_x": "x", "representative_y": "y"}
    )[["trial_id", "trajectory_name", "segment_id", "x", "y"]].copy()
    representative_points["method"] = "segment_representative"

    endpoint_clustered, endpoint_summary = _cluster_method(
        endpoint_points,
        method="endpoint",
        eps_m=args.eps_m,
        min_samples=args.min_samples,
    )
    representative_clustered, representative_summary = _cluster_method(
        representative_points,
        method="segment_representative",
        eps_m=args.eps_m,
        min_samples=args.min_samples,
    )

    selected = pd.concat([endpoint_summary, representative_summary], ignore_index=True)
    selected = selected[selected["is_selected_cluster"]].copy()
    selected["true_center_x"] = TRUE_CENTER[0]
    selected["true_center_y"] = TRUE_CENTER[1]
    selected["x_error_m"] = selected["estimated_x"] - TRUE_CENTER[0]
    selected["y_error_m"] = selected["estimated_y"] - TRUE_CENTER[1]
    selected["true_center_distance_error_m"] = np.hypot(
        selected["x_error_m"], selected["y_error_m"]
    )
    selected["distance_to_other_selected_m"] = np.nan
    if len(selected) == 2:
        coords = selected[["estimated_x", "estimated_y"]].to_numpy(dtype=float)
        distance = float(np.linalg.norm(coords[0] - coords[1]))
        selected["distance_to_other_selected_m"] = distance

    all_summary = pd.concat([endpoint_summary, representative_summary], ignore_index=True)
    all_clustered = pd.concat([endpoint_clustered, representative_clustered], ignore_index=True)

    summary_path = output_dir / "dbscan_estimated_points_summary.csv"
    selected_path = output_dir / "dbscan_selected_estimated_points.csv"
    clustered_points_path = output_dir / "dbscan_clustered_points.csv"
    all_summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)
    all_clustered.to_csv(clustered_points_path, index=False)

    trajectories = _load_trajectories(manifest)
    overview_path = output_dir / "endpoint_vs_segment_representative_dbscan_overview.png"
    zoom_path = output_dir / "endpoint_vs_segment_representative_dbscan_zoom.png"
    _plot_comparison(
        trajectories,
        endpoint_clustered,
        representative_clustered,
        selected,
        overview_path,
        title="DBSCAN estimates from endpoints vs segment representatives",
        limits=_overview_limits(trajectories, endpoint_clustered, representative_clustered),
    )
    _plot_comparison(
        trajectories,
        endpoint_clustered,
        representative_clustered,
        selected,
        zoom_path,
        title="DBSCAN estimates near table A",
        limits=_zoom_limits(endpoint_clustered, representative_clustered),
    )

    print(f"Summary: {summary_path}")
    print(f"Selected estimates: {selected_path}")
    print(f"Clustered points: {clustered_points_path}")
    print(f"Overview figure: {overview_path}")
    print(f"Zoom figure: {zoom_path}")
    print(selected.to_string(index=False))


def _cluster_method(
    points: pd.DataFrame,
    *,
    method: str,
    eps_m: float,
    min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clustered = points.copy().reset_index(drop=True)
    coords = clustered[["x", "y"]].to_numpy(dtype=float)
    labels = _dbscan(coords, eps_m, min_samples)
    clustered["dbscan_label"] = labels
    clustered["dbscan_cluster_id"] = [
        f"{method}_{label}" if label >= 0 else f"{method}_noise" for label in labels
    ]

    rows: list[dict[str, Any]] = []
    non_noise_labels = sorted(label for label in set(labels) if label >= 0)
    selected_label = None
    if non_noise_labels:
        selected_label = max(non_noise_labels, key=lambda label: int((labels == label).sum()))

    for label in sorted(set(labels)):
        mask = labels == label
        label_coords = coords[mask]
        if label == -1:
            cluster_id = f"{method}_noise"
        else:
            cluster_id = f"{method}_{label}"
        centroid = label_coords.mean(axis=0)
        distances = np.linalg.norm(label_coords - centroid, axis=1)
        rows.append(
            {
                "method": method,
                "dbscan_cluster_id": cluster_id,
                "dbscan_label": int(label),
                "point_count": int(mask.sum()),
                "estimated_x": float(centroid[0]),
                "estimated_y": float(centroid[1]),
                "mean_distance_to_estimate_m": float(distances.mean()) if len(distances) else np.nan,
                "max_distance_to_estimate_m": float(distances.max()) if len(distances) else np.nan,
                "eps_m": float(eps_m),
                "min_samples": int(min_samples),
                "is_selected_cluster": bool(label == selected_label),
            }
        )

    return clustered, pd.DataFrame(rows)


def _load_trajectories(manifest: pd.DataFrame) -> list[pd.DataFrame]:
    trajectories = []
    for _, row in manifest.iterrows():
        path = Path(str(row["trajectory_csv"]))
        if not path.exists():
            continue
        trajectory = pd.read_csv(path)
        if trajectory.empty:
            continue
        trajectories.append(trajectory)
    return trajectories


def _plot_comparison(
    trajectories: list[pd.DataFrame],
    endpoint_points: pd.DataFrame,
    representative_points: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8.2))
    for trajectory in trajectories:
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            color="0.70",
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
        )
    _draw_table_a(ax)
    _draw_true_center(ax)
    _draw_points(ax, endpoint_points, color="tab:blue", marker="o", label="endpoints")
    _draw_points(
        ax,
        representative_points,
        color="tab:orange",
        marker="^",
        label="segment representatives",
    )
    _draw_estimate_distance(ax, selected)
    _draw_selected_estimates(ax, selected)
    ax.scatter([0.0], [0.0], marker="s", s=64, color="tab:green", label="start", zorder=10)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    handles = [
        plt.Line2D([0], [0], color="0.70", linewidth=1.0, label="trimmed trajectories"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue", markersize=7, label="endpoints"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="tab:orange", markersize=8, label="segment representatives"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="tab:blue", markeredgecolor="black", markersize=15, label="endpoint estimate"),
        plt.Line2D([0], [0], marker="P", color="none", markerfacecolor="tab:orange", markeredgecolor="black", markersize=13, label="segment estimate"),
        plt.Line2D([0], [0], marker="X", color="none", markerfacecolor="red", markeredgecolor="black", markersize=11, label="true center"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:green", markersize=8, label="start"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_points(
    ax: plt.Axes,
    points: pd.DataFrame,
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    selected = points[points["dbscan_label"] >= 0]
    noise = points[points["dbscan_label"] < 0]
    ax.scatter(
        selected["x"],
        selected["y"],
        s=24,
        marker=marker,
        color=color,
        alpha=0.58,
        edgecolors="white",
        linewidths=0.25,
        label=label,
        zorder=5,
    )
    if not noise.empty:
        ax.scatter(
            noise["x"],
            noise["y"],
            s=38,
            marker="x",
            color=color,
            alpha=0.78,
            linewidths=1.0,
            label=f"{label} noise",
            zorder=6,
        )


def _draw_selected_estimates(ax: plt.Axes, selected: pd.DataFrame) -> None:
    for _, row in selected.iterrows():
        method = str(row["method"])
        x = float(row["estimated_x"])
        y = float(row["estimated_y"])
        if method == "endpoint":
            marker = "*"
            color = "tab:blue"
            text = "endpoint estimate"
            size = 320
            xytext = (10, 14)
            ha = "left"
        else:
            marker = "P"
            color = "tab:orange"
            text = "segment estimate"
            size = 230
            xytext = (-104, -24)
            ha = "left"
        ax.scatter(
            [x],
            [y],
            marker=marker,
            s=size,
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=12,
        )
        ax.annotate(
            text,
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=9,
            color="black",
            weight="bold",
            ha=ha,
            zorder=13,
        )


def _draw_estimate_distance(ax: plt.Axes, selected: pd.DataFrame) -> None:
    if len(selected) != 2:
        return
    coords = selected[["estimated_x", "estimated_y"]].to_numpy(dtype=float)
    distance = float(np.linalg.norm(coords[0] - coords[1]))
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.4,
        alpha=0.80,
        zorder=11,
    )
    midpoint = coords.mean(axis=0)
    ax.annotate(
        f"{distance:.2f} m",
        (float(midpoint[0]), float(midpoint[1])),
        xytext=(6, -16),
        textcoords="offset points",
        fontsize=9,
        color="black",
        zorder=13,
    )


def _draw_true_center(ax: plt.Axes) -> None:
    ax.scatter(
        [TRUE_CENTER[0]],
        [TRUE_CENTER[1]],
        marker="X",
        s=180,
        color="red",
        edgecolor="black",
        linewidth=1.0,
        zorder=14,
    )
    ax.annotate(
        "true center",
        TRUE_CENTER,
        xytext=(8, -22),
        textcoords="offset points",
        fontsize=9,
        color="red",
        weight="bold",
        zorder=15,
    )


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


def _overview_limits(
    trajectories: list[pd.DataFrame],
    endpoint_points: pd.DataFrame,
    representative_points: pd.DataFrame,
) -> tuple[float, float, float, float]:
    xs = [0.0, TABLE_A["left"], TABLE_A["left"] + TABLE_A["width"]]
    ys = [0.0, TABLE_A["bottom"], TABLE_A["bottom"] + TABLE_A["height"]]
    xs.append(TRUE_CENTER[0])
    ys.append(TRUE_CENTER[1])
    for trajectory in trajectories:
        xs.extend(trajectory["x"].to_list())
        ys.extend(trajectory["y"].to_list())
    xs.extend(endpoint_points["x"].to_list())
    ys.extend(endpoint_points["y"].to_list())
    xs.extend(representative_points["x"].to_list())
    ys.extend(representative_points["y"].to_list())
    return _padded_limits(xs, ys, 0.08)


def _zoom_limits(
    endpoint_points: pd.DataFrame,
    representative_points: pd.DataFrame,
) -> tuple[float, float, float, float]:
    xs = [TABLE_A["left"], TABLE_A["left"] + TABLE_A["width"]]
    ys = [TABLE_A["bottom"], TABLE_A["bottom"] + TABLE_A["height"]]
    xs.append(TRUE_CENTER[0])
    ys.append(TRUE_CENTER[1])
    xs.extend(endpoint_points["x"].to_list())
    ys.extend(endpoint_points["y"].to_list())
    xs.extend(representative_points["x"].to_list())
    ys.extend(representative_points["y"].to_list())
    return _padded_limits(xs, ys, 0.10)


def _padded_limits(xs: list[float], ys: list[float], ratio: float) -> tuple[float, float, float, float]:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.55, (x_max - x_min) * ratio)
    pad_y = max(0.55, (y_max - y_min) * ratio)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


if __name__ == "__main__":
    main()
