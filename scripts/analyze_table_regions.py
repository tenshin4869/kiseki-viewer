from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.io import find_trial_dirs, output_trial_id
from pdr_visualizer.plotting import (
    plot_table_region_alpha_shapes,
    plot_table_region_ellipses,
    plot_table_region_classification_clean,
    plot_table_regions,
)
from pdr_visualizer.table_region import (
    add_step_metrics,
    classify_table_area_segments,
    cluster_representative_points,
    destination_label_from_trial_id,
    is_target_trial,
    segment_trajectory,
    select_table_area_points_for_segment,
    summarize_cluster_alpha_shapes,
    summarize_cluster_ellipses,
    summarize_cluster_representatives,
)
from pdr_visualizer.trial import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate table A/C regions from Data001 and Data002 trajectories."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--no-run-missing",
        action="store_true",
        help="Do not run missing trials before analysis.",
    )
    parser.add_argument(
        "--speed-drop-ratio",
        type=float,
        default=None,
        help="Override the speed drop ratio used for seat-area classification.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    table_config = config["table_region"]
    if args.speed_drop_ratio is not None:
        table_config["segmentation"]["speed_drop_ratio"] = args.speed_drop_ratio
    segmentation_config = dict(table_config["segmentation"])
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    output_root = Path(config["paths"]["output_dir"])
    table_output_dir = Path(args.output_dir) if args.output_dir else output_root / "table_regions"
    table_output_dir.mkdir(parents=True, exist_ok=True)

    datasets = set(table_config["datasets"])
    target_labels = set(table_config["target_labels"])
    run_missing = bool(table_config.get("run_missing_trials", True)) and not args.no_run_missing

    target_trials = []
    for trial_dir in find_trial_dirs(raw_data_dir):
        trial_id = str(trial_dir.relative_to(raw_data_dir))
        if is_target_trial(trial_id, datasets, target_labels):
            target_trials.append((trial_id, trial_dir))

    trajectories: list[tuple[str, str, pd.DataFrame]] = []
    all_segments = []
    all_points = []
    all_speed_features = []
    trajectory_by_trial: dict[str, pd.DataFrame] = {}
    for trial_id, trial_dir in target_trials:
        destination_label = destination_label_from_trial_id(trial_id)
        output_id = output_trial_id(raw_data_dir, trial_dir)
        trajectory_path = output_root / output_id / "processed" / "trajectory.csv"
        if run_missing and not trajectory_path.exists():
            run_trial(trial_id, config)
        if not trajectory_path.exists():
            print(f"Skipping missing trajectory CSV: {trajectory_path}")
            continue
        trajectory_df = pd.read_csv(trajectory_path)
        speed_features = add_step_metrics(trajectory_df, segmentation_config)
        speed_features.insert(0, "destination_label", destination_label)
        speed_features.insert(0, "trial_id", trial_id)
        all_speed_features.append(speed_features)
        trajectory_by_trial[trial_id] = speed_features
        trajectories.append((trial_id, destination_label, trajectory_df))
        segments_df, points_df = segment_trajectory(
            trial_id,
            destination_label,
            trajectory_df,
            segmentation_config,
        )
        all_segments.append(segments_df)
        all_points.append(points_df)

    segments = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    segments = classify_table_area_segments(segments, segmentation_config)
    speed_features = (
        pd.concat(all_speed_features, ignore_index=True) if all_speed_features else pd.DataFrame()
    )
    segment_summary = _summarize_segments_by_trial(segments)
    speed_summary = _summarize_speed_features(speed_features)
    seat_area_points = _seat_area_segment_points(segments, trajectory_by_trial)
    points = seat_area_points.copy()
    seat_area_segment_representatives = _seat_area_segment_representatives(segments)
    clustered_representative_points = cluster_representative_points(
        points,
        eps_m=float(table_config["clustering"]["dbscan_eps_m"]),
        min_samples=int(table_config["clustering"]["dbscan_min_samples"]),
    )
    clustered_seat_area_points = cluster_representative_points(
        seat_area_points,
        eps_m=float(table_config["clustering"]["dbscan_eps_m"]),
        min_samples=int(table_config["clustering"]["dbscan_min_samples"]),
    )
    clustered_seat_area_segment_representatives = cluster_representative_points(
        seat_area_segment_representatives,
        eps_m=float(table_config["clustering"]["dbscan_eps_m"]),
        min_samples=int(table_config["clustering"]["dbscan_min_samples"]),
    )
    table_points = summarize_cluster_representatives(clustered_seat_area_points)
    table_points_from_segment_representatives = summarize_cluster_representatives(
        clustered_seat_area_segment_representatives
    )
    alpha_shape_source_segments = _alpha_shape_source_segments(
        segments,
        clustered_seat_area_segment_representatives,
    )
    table_region_ellipses = summarize_cluster_ellipses(
        clustered_seat_area_segment_representatives,
        n_std=2.0,
    )
    alpha_shape_values = [0.9, 1.5]
    table_region_alpha_shapes_by_alpha = {
        alpha: summarize_cluster_alpha_shapes(
            clustered_seat_area_segment_representatives,
            alpha=alpha,
            min_points=4,
        )
        for alpha in alpha_shape_values
    }

    segments_path = table_output_dir / "segments.csv"
    segment_summary_path = table_output_dir / "segment_summary.csv"
    speed_features_path = table_output_dir / "trajectory_speed_features.csv"
    speed_summary_path = table_output_dir / "speed_feature_summary.csv"
    seat_area_points_path = table_output_dir / "seat_area_segment_points.csv"
    seat_area_segment_representatives_path = table_output_dir / "seat_area_segment_representatives.csv"
    points_path = table_output_dir / "representative_points.csv"
    clustered_representative_points_path = table_output_dir / "clustered_representative_points.csv"
    clustered_seat_area_points_path = table_output_dir / "clustered_seat_area_segment_points.csv"
    clustered_seat_area_segment_representatives_path = (
        table_output_dir / "clustered_seat_area_segment_representatives.csv"
    )
    table_points_path = table_output_dir / "table_region_points.csv"
    table_points_from_segment_representatives_path = (
        table_output_dir / "table_region_points_from_segment_representatives.csv"
    )
    alpha_shape_source_segments_path = table_output_dir / "table_region_alpha_shape_source_segments.csv"
    table_region_ellipses_path = table_output_dir / "table_region_ellipses_from_segment_centers.csv"
    table_region_alpha_shapes_path = table_output_dir / "table_region_alpha_shapes_from_segment_centers.csv"
    table_region_alpha_shape_paths = {
        0.9: table_region_alpha_shapes_path,
        1.5: table_output_dir / "table_region_alpha_shapes_from_segment_centers_alpha1p5.csv",
    }
    figure_path = table_output_dir / "table_region_cluster_points.png"
    segment_representative_figure_path = table_output_dir / "table_region_segment_representatives.png"
    ellipse_figure_path = table_output_dir / "table_region_ellipses_from_segment_centers.png"
    alpha_shape_figure_path = table_output_dir / "table_region_alpha_shapes_from_segment_centers.png"
    alpha_shape_figure_paths = {
        0.9: alpha_shape_figure_path,
        1.5: table_output_dir / "table_region_alpha_shapes_from_segment_centers_alpha1p5.png",
    }
    segments.to_csv(segments_path, index=False)
    segment_summary.to_csv(segment_summary_path, index=False)
    speed_features.to_csv(speed_features_path, index=False)
    speed_summary.to_csv(speed_summary_path, index=False)
    seat_area_points.to_csv(seat_area_points_path, index=False)
    seat_area_segment_representatives.to_csv(seat_area_segment_representatives_path, index=False)
    points.to_csv(points_path, index=False)
    clustered_representative_points.to_csv(clustered_representative_points_path, index=False)
    clustered_seat_area_points.to_csv(clustered_seat_area_points_path, index=False)
    clustered_seat_area_segment_representatives.to_csv(
        clustered_seat_area_segment_representatives_path, index=False
    )
    table_points.to_csv(table_points_path, index=False)
    table_points_from_segment_representatives.to_csv(
        table_points_from_segment_representatives_path, index=False
    )
    alpha_shape_source_segments.to_csv(alpha_shape_source_segments_path, index=False)
    table_region_ellipses.to_csv(table_region_ellipses_path, index=False)
    for alpha, alpha_shapes in table_region_alpha_shapes_by_alpha.items():
        alpha_shapes.to_csv(table_region_alpha_shape_paths[alpha], index=False)
    plot_table_regions(
        trajectories,
        segments,
        clustered_seat_area_points,
        table_points,
        table_config.get("table_overlays", {}),
        figure_path,
        "Data001/Data002 table-region cluster points",
        dpi=int(config["visualization"]["figure_dpi"]),
        equal_axis=bool(config["visualization"]["equal_axis"]),
        show_grid=bool(config["visualization"]["show_grid"]),
    )
    plot_table_region_classification_clean(
        trajectories,
        segments,
        table_points_from_segment_representatives,
        table_config.get("table_overlays", {}),
        segment_representative_figure_path,
        "Data001/Data002 segment classification with representatives",
        dpi=int(config["visualization"]["figure_dpi"]),
        equal_axis=bool(config["visualization"]["equal_axis"]),
        show_grid=bool(config["visualization"]["show_grid"]),
    )
    plot_table_region_ellipses(
        trajectories,
        segments,
        clustered_seat_area_segment_representatives,
        table_region_ellipses,
        table_config.get("table_overlays", {}),
        ellipse_figure_path,
        "Data001/Data002 ellipse regions from segment-center clusters",
        dpi=int(config["visualization"]["figure_dpi"]),
        equal_axis=bool(config["visualization"]["equal_axis"]),
        show_grid=bool(config["visualization"]["show_grid"]),
    )
    for alpha in alpha_shape_values:
        plot_table_region_alpha_shapes(
            clustered_seat_area_segment_representatives,
            alpha=alpha,
            min_points=4,
            table_overlays=table_config.get("table_overlays", {}),
            output_path=alpha_shape_figure_paths[alpha],
            title=f"Data001/Data002 alpha-shape regions from segment-center clusters alpha={alpha}",
            dpi=int(config["visualization"]["figure_dpi"]),
            equal_axis=bool(config["visualization"]["equal_axis"]),
            show_grid=bool(config["visualization"]["show_grid"]),
        )
    print(f"Target trajectories: {len(trajectories)}")
    print(f"Segments: {segments_path}")
    print(f"Segment summary: {segment_summary_path}")
    print(f"Speed features: {speed_features_path}")
    print(f"Speed feature summary: {speed_summary_path}")
    print(f"Seat-area segment points: {seat_area_points_path}")
    print(f"Seat-area segment representatives: {seat_area_segment_representatives_path}")
    print(f"Representative points: {points_path}")
    print(f"Clustered representative points: {clustered_representative_points_path}")
    print(f"Clustered seat-area points: {clustered_seat_area_points_path}")
    print(f"Clustered seat-area segment representatives: {clustered_seat_area_segment_representatives_path}")
    print(f"Estimated table points: {table_points_path}")
    print(f"Estimated table points from segment representatives: {table_points_from_segment_representatives_path}")
    print(f"Alpha-shape source segments: {alpha_shape_source_segments_path}")
    print(f"Ellipse regions from segment centers: {table_region_ellipses_path}")
    for alpha in alpha_shape_values:
        print(f"Alpha-shape regions from segment centers alpha={alpha}: {table_region_alpha_shape_paths[alpha]}")
    print(f"Representative point figure: {figure_path}")
    print(f"Segment representative figure: {segment_representative_figure_path}")
    print(f"Ellipse region figure: {ellipse_figure_path}")
    for alpha in alpha_shape_values:
        print(f"Alpha-shape region figure alpha={alpha}: {alpha_shape_figure_paths[alpha]}")


def _summarize_segments_by_trial(segments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "destination_label",
        "segment_count",
        "corridor_segment_count",
        "table_area_segment_count",
        "segment_ranges",
    ]
    if segments.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (trial_id, label), group in segments.groupby(["trial_id", "destination_label"], sort=True):
        ordered = group.sort_values("segment_id")
        segment_ranges = []
        for _, row in ordered.iterrows():
            segment_ranges.append(
                f"{int(row['segment_id'])}:"
                f"{int(row['start_step_index'])}-{int(row['end_step_index'])}"
                f"({row['classification']})"
            )
        rows.append(
            {
                "trial_id": trial_id,
                "destination_label": label,
                "segment_count": int(len(ordered)),
                "corridor_segment_count": int((ordered["classification"] == "corridor").sum()),
                "table_area_segment_count": int((ordered["classification"] == "seat_area").sum()),
                "segment_ranges": "; ".join(segment_ranges),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _seat_area_segment_points(
    segments: pd.DataFrame, trajectory_by_trial: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows = []
    if segments.empty:
        return pd.DataFrame(
            columns=[
                "trial_id",
                "destination_label",
                "segment_id",
                "step_index",
                "x",
                "y",
                "step_time",
                "speed_mps",
                "speed_drop_ratio",
                "speed_change_ratio",
                "is_slowdown",
                "is_low_speed",
            ]
        )

    seat_segments = segments[segments["classification"] == "seat_area"]
    for _, segment in seat_segments.iterrows():
        trial_id = str(segment["trial_id"])
        trajectory = trajectory_by_trial[trial_id]
        start = int(segment["start_row"])
        end = int(segment["end_row"])
        selected = select_table_area_points_for_segment(trajectory.iloc[start:end])
        for _, point in selected.iterrows():
            rows.append(
                {
                    "trial_id": trial_id,
                    "destination_label": str(segment["destination_label"]),
                    "segment_id": int(segment["segment_id"]),
                    "step_index": int(point["step_index"]),
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                    "step_time": float(point["step_time"]),
                    "speed_mps": float(point["speed_mps"]),
                    "speed_drop_ratio": float(point["speed_drop_ratio"]),
                    "speed_change_ratio": float(point["speed_change_ratio"]),
                    "is_slowdown": bool(point["is_slowdown"]),
                    "is_low_speed": bool(point["is_low_speed"]),
                }
            )
    return pd.DataFrame(rows)


def _seat_area_segment_representatives(segments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "destination_label",
        "segment_id",
        "x",
        "y",
        "step_count",
        "start_step_index",
        "end_step_index",
    ]
    if segments.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    seat_segments = segments[segments["classification"] == "seat_area"]
    for _, segment in seat_segments.iterrows():
        rows.append(
            {
                "trial_id": str(segment["trial_id"]),
                "destination_label": str(segment["destination_label"]),
                "segment_id": int(segment["segment_id"]),
                "x": float(segment.get("segment_center_x", segment["representative_x"])),
                "y": float(segment.get("segment_center_y", segment["representative_y"])),
                "step_count": int(segment["step_count"]),
                "start_step_index": int(segment["start_step_index"]),
                "end_step_index": int(segment["end_step_index"]),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _alpha_shape_source_segments(
    segments: pd.DataFrame,
    clustered_segment_representatives: pd.DataFrame,
) -> pd.DataFrame:
    if segments.empty or clustered_segment_representatives.empty:
        return pd.DataFrame()

    selected_frames = []
    for _, group in clustered_segment_representatives.groupby("destination_label", sort=True):
        non_noise = group[group["cluster_id"] != "noise"]
        if not non_noise.empty:
            cluster_id = non_noise.groupby("cluster_id").size().sort_values(ascending=False).index[0]
            selected = non_noise[non_noise["cluster_id"] == cluster_id]
        else:
            selected = group
        selected_frames.append(selected)

    selected_points = pd.concat(selected_frames, ignore_index=True)
    selected_points = selected_points[
        ["trial_id", "destination_label", "segment_id", "cluster_id", "x", "y"]
    ].rename(columns={"x": "alpha_shape_source_x", "y": "alpha_shape_source_y"})

    merged = selected_points.merge(
        segments,
        on=["trial_id", "destination_label", "segment_id"],
        how="left",
    )
    preferred_columns = [
        "trial_id",
        "destination_label",
        "segment_id",
        "cluster_id",
        "classification",
        "alpha_shape_source_x",
        "alpha_shape_source_y",
        "segment_center_x",
        "segment_center_y",
        "representative_x",
        "representative_y",
        "start_step_index",
        "end_step_index",
        "step_count",
        "path_length_m",
        "mean_speed_mps",
        "slowdown_ratio",
        "low_speed_ratio",
        "heading_change_deg",
        "has_speed_table_evidence",
        "is_near_speed_table_segment",
    ]
    columns = [column for column in preferred_columns if column in merged.columns]
    remaining = [column for column in merged.columns if column not in columns]
    return merged[columns + remaining].sort_values(
        ["destination_label", "trial_id", "segment_id"]
    )


def _summarize_speed_features(speed_features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trial_id",
        "destination_label",
        "step_count",
        "speed_mean_mps",
        "speed_std_mps",
        "speed_p25_mps",
        "speed_p50_mps",
        "speed_p75_mps",
        "speed_drop_p50",
        "speed_drop_p75",
        "speed_drop_p90",
        "speed_change_p50",
        "speed_change_p75",
        "speed_change_p90",
        "heading_delta_p50_deg",
        "heading_delta_p75_deg",
        "heading_delta_p90_deg",
        "slowdown_point_count",
        "slowdown_point_ratio",
        "low_speed_point_count",
        "low_speed_point_ratio",
    ]
    if speed_features.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (trial_id, label), group in speed_features.groupby(
        ["trial_id", "destination_label"], sort=True
    ):
        speed = group["speed_mps"]
        speed_drop = group["speed_drop_ratio"]
        speed_change = group["speed_change_ratio"]
        heading_delta = group["heading_delta_deg"]
        rows.append(
            {
                "trial_id": trial_id,
                "destination_label": label,
                "step_count": len(group),
                "speed_mean_mps": float(speed.mean()),
                "speed_std_mps": float(speed.std(ddof=0)),
                "speed_p25_mps": float(speed.quantile(0.25)),
                "speed_p50_mps": float(speed.quantile(0.50)),
                "speed_p75_mps": float(speed.quantile(0.75)),
                "speed_drop_p50": float(speed_drop.quantile(0.50)),
                "speed_drop_p75": float(speed_drop.quantile(0.75)),
                "speed_drop_p90": float(speed_drop.quantile(0.90)),
                "speed_change_p50": float(speed_change.quantile(0.50)),
                "speed_change_p75": float(speed_change.quantile(0.75)),
                "speed_change_p90": float(speed_change.quantile(0.90)),
                "heading_delta_p50_deg": float(heading_delta.quantile(0.50)),
                "heading_delta_p75_deg": float(heading_delta.quantile(0.75)),
                "heading_delta_p90_deg": float(heading_delta.quantile(0.90)),
                "slowdown_point_count": int(group["is_slowdown"].sum()),
                "slowdown_point_ratio": float(group["is_slowdown"].mean()),
                "low_speed_point_count": int(group["is_low_speed"].sum()),
                "low_speed_point_ratio": float(group["is_low_speed"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    main()
