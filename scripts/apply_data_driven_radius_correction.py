from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply data-driven radius correction using segment-representative DBSCAN cluster spread."
    )
    parser.add_argument("--manifest", default="output/a_trajectories_first_stop_filtered/manifest.csv")
    parser.add_argument(
        "--clustered-points",
        default="output/a_dbscan_estimate_comparison_first_stop_filtered/dbscan_clustered_points.csv",
    )
    parser.add_argument("--output-dir", default="output/a_radius_corrected_trajectories_segment_representative")
    parser.add_argument("--radius-quantile", type=float, default=0.80)
    parser.add_argument("--progress-power", type=float, default=2.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    trials_dir = output_dir / "trials"
    figures_dir = output_dir / "figures"
    trials_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    anchor, radius, cluster_points, radius_summary = _estimate_anchor_and_radius(
        Path(args.clustered_points),
        args.radius_quantile,
    )
    corrected_manifest, correction_rows, trajectory_pairs = _correct_trajectories(
        manifest,
        anchor,
        radius,
        args.progress_power,
        trials_dir,
    )

    corrected_manifest_path = output_dir / "manifest.csv"
    correction_summary_path = output_dir / "radius_correction_summary.csv"
    radius_summary_path = output_dir / "radius_definition_summary.csv"
    cluster_points_path = output_dir / "radius_source_segment_representative_points.csv"

    corrected_manifest.to_csv(corrected_manifest_path, index=False)
    pd.DataFrame(correction_rows).to_csv(correction_summary_path, index=False)
    radius_summary.to_csv(radius_summary_path, index=False)
    cluster_points.to_csv(cluster_points_path, index=False)

    correction_summary = pd.DataFrame(correction_rows)
    _plot_overall(
        trajectory_pairs,
        anchor,
        radius,
        figures_dir / "radius_correction_trajectories_overall.png",
    )
    _plot_grid(
        trajectory_pairs,
        anchor,
        radius,
        figures_dir / "radius_correction_trajectories_by_seat_2x2.png",
    )
    _plot_endpoint_shift(
        correction_summary,
        anchor,
        radius,
        figures_dir / "radius_correction_endpoint_shift.png",
    )
    _plot_individual_examples(
        trajectory_pairs,
        correction_summary,
        anchor,
        radius,
        figures_dir / "radius_correction_individual_examples.png",
    )

    print(f"Anchor: ({anchor[0]:.6f}, {anchor[1]:.6f})")
    print(f"Radius q={args.radius_quantile:g}: {radius:.6f} m")
    print(f"Manifest: {corrected_manifest_path}")
    print(f"Correction summary: {correction_summary_path}")
    print(f"Radius summary: {radius_summary_path}")
    print(f"Overall figure: {figures_dir / 'radius_correction_trajectories_overall.png'}")
    print(f"2x2 figure: {figures_dir / 'radius_correction_trajectories_by_seat_2x2.png'}")
    print(f"Endpoint shift figure: {figures_dir / 'radius_correction_endpoint_shift.png'}")
    print(f"Individual examples: {figures_dir / 'radius_correction_individual_examples.png'}")


def _estimate_anchor_and_radius(
    clustered_points_path: Path,
    radius_quantile: float,
) -> tuple[np.ndarray, float, pd.DataFrame, pd.DataFrame]:
    clustered = pd.read_csv(clustered_points_path)
    points = clustered[
        (clustered["method"] == "segment_representative")
        & (clustered["dbscan_label"] != -1)
    ].copy()
    if points.empty:
        raise ValueError("No selected segment-representative DBSCAN cluster points were found.")

    cluster_sizes = points.groupby("dbscan_label").size().sort_values(ascending=False)
    selected_label = cluster_sizes.index[0]
    points = points[points["dbscan_label"] == selected_label].copy()
    anchor = points[["x", "y"]].mean().to_numpy(dtype=float)
    distances = np.linalg.norm(points[["x", "y"]].to_numpy(dtype=float) - anchor, axis=1)
    radius = float(np.quantile(distances, radius_quantile))
    points["distance_to_anchor_m"] = distances

    summary = pd.DataFrame(
        [
            {
                "method": "segment_representative_dbscan_cluster_spread",
                "dbscan_label": int(selected_label),
                "cluster_point_count": int(len(points)),
                "anchor_x": float(anchor[0]),
                "anchor_y": float(anchor[1]),
                "radius_quantile": float(radius_quantile),
                "radius_m": radius,
                "distance_mean_m": float(distances.mean()),
                "distance_median_m": float(np.median(distances)),
                "distance_q75_m": float(np.quantile(distances, 0.75)),
                "distance_q80_m": float(np.quantile(distances, 0.80)),
                "distance_q90_m": float(np.quantile(distances, 0.90)),
                "distance_max_m": float(distances.max()),
            }
        ]
    )
    return anchor, radius, points, summary


def _correct_trajectories(
    manifest: pd.DataFrame,
    anchor: np.ndarray,
    radius: float,
    progress_power: float,
    trials_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[dict[str, object]]]:
    manifest_rows: list[dict[str, object]] = []
    correction_rows: list[dict[str, object]] = []
    trajectory_pairs: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        original_path = Path(row["trajectory_csv"])
        trajectory = pd.read_csv(original_path)
        corrected = trajectory.copy()
        endpoint = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        vector = endpoint - anchor
        distance_before = float(np.linalg.norm(vector))

        if distance_before > radius and distance_before > 0:
            corrected_endpoint = anchor + radius * vector / distance_before
            endpoint_shift = corrected_endpoint - endpoint
        else:
            corrected_endpoint = endpoint.copy()
            endpoint_shift = np.zeros(2, dtype=float)

        progress = _progress_values(trajectory)
        weights = progress**progress_power
        corrected["x"] = trajectory["x"].to_numpy(dtype=float) + weights * endpoint_shift[0]
        corrected["y"] = trajectory["y"].to_numpy(dtype=float) + weights * endpoint_shift[1]
        corrected_endpoint_actual = corrected[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        distance_after = float(np.linalg.norm(corrected_endpoint_actual - anchor))
        shift_distance = float(np.linalg.norm(endpoint_shift))

        relative_path = _relative_trial_path(row)
        output_path = trials_dir / relative_path / "trajectory.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(output_path, index=False)

        manifest_row = row.to_dict()
        manifest_row["trajectory_csv"] = str(output_path)
        manifest_row["endpoint_x"] = float(corrected_endpoint_actual[0])
        manifest_row["endpoint_y"] = float(corrected_endpoint_actual[1])
        manifest_row["processing_mode"] = f"{row.get('processing_mode', '')}+data_driven_radius_correction"
        manifest_rows.append(manifest_row)

        correction_rows.append(
            {
                "trial_id": row["trial_id"],
                "trajectory_name": row["trajectory_name"],
                "original_trajectory_csv": str(original_path),
                "corrected_trajectory_csv": str(output_path),
                "original_endpoint_x": float(endpoint[0]),
                "original_endpoint_y": float(endpoint[1]),
                "corrected_endpoint_x": float(corrected_endpoint_actual[0]),
                "corrected_endpoint_y": float(corrected_endpoint_actual[1]),
                "distance_to_anchor_before_m": distance_before,
                "distance_to_anchor_after_m": distance_after,
                "endpoint_shift_m": shift_distance,
                "endpoint_shift_x": float(endpoint_shift[0]),
                "endpoint_shift_y": float(endpoint_shift[1]),
                "was_corrected": bool(shift_distance > 1e-12),
                "radius_m": float(radius),
                "progress_power": float(progress_power),
            }
        )
        trajectory_pairs.append(
            {
                "trial_id": row["trial_id"],
                "trajectory_name": row["trajectory_name"],
                "before": trajectory,
                "after": corrected,
                "endpoint_shift_m": shift_distance,
            }
        )

    return pd.DataFrame(manifest_rows), correction_rows, trajectory_pairs


def _progress_values(trajectory: pd.DataFrame) -> np.ndarray:
    if "step_index" in trajectory.columns:
        values = trajectory["step_index"].to_numpy(dtype=float)
        if values.max() > values.min():
            return (values - values.min()) / (values.max() - values.min())
    if len(trajectory) <= 1:
        return np.ones(len(trajectory), dtype=float)
    return np.linspace(0.0, 1.0, len(trajectory))


def _relative_trial_path(row: pd.Series) -> Path:
    trial_id = str(row["trial_id"]).strip("/")
    if trial_id:
        return Path(trial_id)
    return Path(str(row["trajectory_name"])) / Path(str(row.name))


def _plot_overall(
    trajectory_pairs: list[dict[str, object]],
    anchor: np.ndarray,
    radius: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    _plot_pairs(ax, trajectory_pairs, show_labels=True)
    _draw_anchor_radius(ax, anchor, radius)
    _finish_axis(ax, "All trajectories: before/after data-driven radius correction")
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(
    trajectory_pairs: list[dict[str, object]],
    anchor: np.ndarray,
    radius: float,
    output_path: Path,
) -> None:
    seats = sorted({str(pair["trajectory_name"]) for pair in trajectory_pairs})
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, seat in zip(axes.ravel(), seats):
        pairs = [pair for pair in trajectory_pairs if pair["trajectory_name"] == seat]
        _plot_pairs(ax, pairs, show_labels=False)
        _draw_anchor_radius(ax, anchor, radius)
        ax.set_title(f"{seat}  n={len(pairs)}")
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")
    handles = [
        plt.Line2D([0], [0], color="0.72", linewidth=1.0, label="before"),
        plt.Line2D([0], [0], color="black", linewidth=1.7, label="after"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="estimated anchor"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="allowed radius"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    fig.suptitle("Data-driven radius correction by A-seat", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_shift(
    summary: pd.DataFrame,
    anchor: np.ndarray,
    radius: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for seat, group in summary.groupby("trajectory_name", sort=True):
        color = COLORS.get(str(seat), "tab:gray")
        ax.scatter(
            group["original_endpoint_x"],
            group["original_endpoint_y"],
            s=26,
            color=color,
            alpha=0.25,
            edgecolors="none",
            label=f"{seat} before",
        )
        ax.scatter(
            group["corrected_endpoint_x"],
            group["corrected_endpoint_y"],
            s=32,
            color=color,
            alpha=0.80,
            edgecolors="white",
            linewidths=0.3,
            label=f"{seat} after",
        )
        for _, row in group[group["was_corrected"]].iterrows():
            ax.plot(
                [row["original_endpoint_x"], row["corrected_endpoint_x"]],
                [row["original_endpoint_y"], row["corrected_endpoint_y"]],
                color=color,
                alpha=0.28,
                linewidth=0.8,
            )
    _draw_anchor_radius(ax, anchor, radius)
    _finish_axis(ax, "Endpoint movement by radius correction")
    ax.legend(loc="upper right", fontsize="x-small", ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_individual_examples(
    trajectory_pairs: list[dict[str, object]],
    summary: pd.DataFrame,
    anchor: np.ndarray,
    radius: float,
    output_path: Path,
) -> None:
    examples = summary.sort_values("endpoint_shift_m", ascending=False).head(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    by_trial = {str(pair["trial_id"]): pair for pair in trajectory_pairs}
    for ax, (_, row) in zip(axes.ravel(), examples.iterrows()):
        pair = by_trial[str(row["trial_id"])]
        _plot_pairs(ax, [pair], show_labels=True)
        _draw_anchor_radius(ax, anchor, radius)
        ax.set_title(
            f"{row['trajectory_name']}  shift={row['endpoint_shift_m']:.2f} m\n{row['trial_id']}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.22)
        ax.set_aspect("equal", adjustable="box")
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    fig.suptitle("Largest individual trajectory changes", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pairs(ax: plt.Axes, trajectory_pairs: list[dict[str, object]], show_labels: bool) -> None:
    before_label_used = False
    after_label_used = False
    for pair in trajectory_pairs:
        before = pair["before"]
        after = pair["after"]
        seat = str(pair["trajectory_name"])
        color = COLORS.get(seat, "tab:gray")
        ax.plot(
            before["x"],
            before["y"],
            color=color,
            alpha=0.16,
            linewidth=1.0,
            label="before" if show_labels and not before_label_used else None,
        )
        ax.plot(
            after["x"],
            after["y"],
            color=color,
            alpha=0.82,
            linewidth=1.45,
            label="after" if show_labels and not after_label_used else None,
        )
        before_label_used = True
        after_label_used = True


def _draw_anchor_radius(ax: plt.Axes, anchor: np.ndarray, radius: float) -> None:
    circle = plt.Circle(anchor, radius, fill=False, color="black", linestyle="--", linewidth=1.4, alpha=0.75)
    ax.add_patch(circle)
    ax.scatter(
        [anchor[0]],
        [anchor[1]],
        marker="*",
        s=250,
        color="gold",
        edgecolor="black",
        linewidth=1.0,
        zorder=8,
    )


def _finish_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


if __name__ == "__main__":
    main()
