from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


@dataclass(frozen=True)
class AlphaShapeResult:
    triangles: np.ndarray
    area_m2: float


def destination_label_from_trial_id(trial_id: str) -> str:
    name = Path(trial_id).name
    return name.split("-", 1)[0]


def is_target_trial(trial_id: str, datasets: set[str], target_labels: set[str]) -> bool:
    path = Path(trial_id)
    return len(path.parts) >= 2 and path.parts[0] in datasets and destination_label_from_trial_id(trial_id) in target_labels


def add_step_metrics(trajectory_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if trajectory_df.empty:
        return trajectory_df.copy()

    df = trajectory_df.copy().reset_index(drop=True)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    t = df["step_time"].to_numpy(dtype=float)
    heading = df["heading_rad"].to_numpy(dtype=float)
    step_length = df["step_length_m"].to_numpy(dtype=float)

    dt = np.diff(np.r_[t[0], t])
    positive_dt = dt[dt > 0]
    fallback_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0
    dt = np.where(dt > 0, dt, fallback_dt)
    speed = step_length / dt
    if len(speed) > 1:
        speed[0] = speed[1]

    heading_delta = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        heading_delta[1:] = np.abs(_angle_differences(heading[1:], heading[:-1]))

    window = int(config.get("speed_window_steps", 4))
    previous_speed = _previous_rolling_mean(speed, window)
    future_speed = _future_rolling_mean(speed, window)
    speed_drop_ratio = np.maximum((previous_speed - speed) / np.maximum(previous_speed, 1e-9), 0.0)
    signed_speed_change_ratio = (future_speed - previous_speed) / np.maximum(previous_speed, 1e-9)
    speed_change_ratio = np.abs(signed_speed_change_ratio)

    progress = np.linspace(1.0 / len(df), 1.0, len(df))
    dist_to_endpoint = np.hypot(x[-1] - x, y[-1] - y)
    slowdown = speed_drop_ratio >= float(config.get("speed_drop_ratio", 0.25))
    low_speed = speed <= previous_speed * float(config.get("low_speed_ratio", 0.65))
    slowdown_turn = _slowdown_followed_by_turn(
        slowdown,
        heading_delta,
        int(config.get("slowdown_turn_window_steps", 3)),
        np.deg2rad(float(config.get("slowdown_turn_deg", 25.0))),
    )
    df["dt_s"] = dt
    df["speed_mps"] = speed
    df["previous_speed_mps"] = previous_speed
    df["future_speed_mps"] = future_speed
    df["signed_speed_change_ratio"] = signed_speed_change_ratio
    df["speed_change_ratio"] = speed_change_ratio
    df["speed_drop_ratio"] = speed_drop_ratio
    df["heading_delta_rad"] = heading_delta
    df["heading_delta_deg"] = np.degrees(heading_delta)
    df["progress_ratio"] = progress
    df["distance_to_endpoint_m"] = dist_to_endpoint
    df["is_slowdown"] = slowdown
    df["is_low_speed"] = low_speed
    df["slowdown_followed_by_turn"] = slowdown_turn
    return df


def segment_trajectory(
    trial_id: str,
    destination_label: str,
    trajectory_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = add_step_metrics(trajectory_df, config)
    if metrics.empty:
        return _empty_segments(), _empty_points()

    turn_boundary = np.deg2rad(float(config.get("turn_boundary_deg", 45.0)))
    turn_window_steps = max(int(config.get("turn_window_steps", 1)), 1)
    speed_change_boundary = float(config.get("speed_change_ratio", 0.25))
    min_boundary_gap = int(config.get("min_boundary_gap_steps", 3))
    boundaries = [0]
    heading_delta = metrics["heading_delta_rad"].to_numpy(dtype=float)
    speed_change_ratio = metrics["speed_change_ratio"].to_numpy(dtype=float)
    for i in range(1, len(metrics)):
        turn_start = max(1, i - turn_window_steps + 1)
        has_turn_change = heading_delta[turn_start : i + 1].sum() >= turn_boundary
        has_speed_change = speed_change_ratio[i] >= speed_change_boundary
        far_enough_from_previous = i - boundaries[-1] >= min_boundary_gap
        if (has_turn_change or has_speed_change) and far_enough_from_previous:
            if i > boundaries[-1]:
                boundaries.append(i)
    if boundaries[-1] != len(metrics):
        boundaries.append(len(metrics))

    min_steps = int(config.get("min_segment_steps", 2))
    intervals = _merge_short_intervals(
        [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:])],
        min_steps,
    )

    rows: list[dict[str, Any]] = []
    for segment_id, (start, end) in enumerate(intervals):
        segment = metrics.iloc[start:end].copy()
        if segment.empty:
            continue
        row = _segment_features(trial_id, destination_label, segment_id, start, end, segment, config)
        rows.append(row)

    segments = _classify_segments(pd.DataFrame(rows), config)
    point_rows: list[dict[str, Any]] = []
    for _, row in segments.iterrows():
        segment = metrics.iloc[int(row["start_row"]):int(row["end_row"])].copy()
        point_rows.extend(_representative_point_rows(row.to_dict(), segment))
    points = pd.DataFrame(point_rows)
    return segments, points


def cluster_representative_points(points_df: pd.DataFrame, eps_m: float, min_samples: int) -> pd.DataFrame:
    if points_df.empty:
        clustered = points_df.copy()
        clustered["cluster_id"] = []
        return clustered

    frames = []
    for label, group in points_df.groupby("destination_label", sort=True):
        coords = group[["x", "y"]].to_numpy(dtype=float)
        labels = _dbscan(coords, eps_m, min_samples)
        clustered = group.copy()
        clustered["cluster_id"] = [f"{label}_{value}" if value >= 0 else "noise" for value in labels]
        frames.append(clustered)
    return pd.concat(frames, ignore_index=True)


def summarize_cluster_representatives(clustered_points: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "destination_label",
        "selected_cluster_id",
        "point_count",
        "centroid_x",
        "centroid_y",
        "std_x",
        "std_y",
        "mean_distance_to_centroid_m",
        "max_distance_to_centroid_m",
        "noise_point_count",
    ]
    if clustered_points.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for label, label_points in clustered_points.groupby("destination_label", sort=True):
        non_noise = label_points[label_points["cluster_id"] != "noise"]
        noise_count = int((label_points["cluster_id"] == "noise").sum())
        if non_noise.empty:
            selected = label_points
            cluster_id = "all_points_no_cluster"
        else:
            cluster_counts = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
            cluster_id = str(cluster_counts.index[0])
            selected = non_noise[non_noise["cluster_id"] == cluster_id]

        coords = selected[["x", "y"]].to_numpy(dtype=float)
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords - centroid, axis=1)
        rows.append(
            {
                "destination_label": label,
                "selected_cluster_id": cluster_id,
                "point_count": len(selected),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "std_x": float(np.std(coords[:, 0])),
                "std_y": float(np.std(coords[:, 1])),
                "mean_distance_to_centroid_m": float(np.mean(distances)),
                "max_distance_to_centroid_m": float(np.max(distances)),
                "noise_point_count": noise_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_cluster_ellipses(clustered_points: pd.DataFrame, n_std: float = 2.0) -> pd.DataFrame:
    columns = [
        "destination_label",
        "selected_cluster_id",
        "point_count",
        "center_x",
        "center_y",
        "var_x",
        "var_y",
        "cov_xy",
        "semi_major_m",
        "semi_minor_m",
        "angle_deg",
        "area_m2",
        "n_std",
        "noise_point_count",
    ]
    if clustered_points.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for label, label_points in clustered_points.groupby("destination_label", sort=True):
        non_noise = label_points[label_points["cluster_id"] != "noise"]
        noise_count = int((label_points["cluster_id"] == "noise").sum())
        if non_noise.empty:
            selected = label_points
            cluster_id = "all_points_no_cluster"
        else:
            cluster_counts = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
            cluster_id = str(cluster_counts.index[0])
            selected = non_noise[non_noise["cluster_id"] == cluster_id]

        coords = selected[["x", "y"]].to_numpy(dtype=float)
        center = coords.mean(axis=0)
        if len(coords) >= 2:
            covariance = np.cov(coords, rowvar=False)
        else:
            covariance = np.zeros((2, 2), dtype=float)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]
        semi_major = float(n_std * np.sqrt(eigenvalues[0]))
        semi_minor = float(n_std * np.sqrt(eigenvalues[1]))
        angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))

        rows.append(
            {
                "destination_label": label,
                "selected_cluster_id": cluster_id,
                "point_count": len(selected),
                "center_x": float(center[0]),
                "center_y": float(center[1]),
                "var_x": float(covariance[0, 0]),
                "var_y": float(covariance[1, 1]),
                "cov_xy": float(covariance[0, 1]),
                "semi_major_m": semi_major,
                "semi_minor_m": semi_minor,
                "angle_deg": angle,
                "area_m2": float(np.pi * semi_major * semi_minor),
                "n_std": float(n_std),
                "noise_point_count": noise_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def alpha_shape(points: np.ndarray, alpha: float, min_points: int = 4) -> AlphaShapeResult:
    points = np.asarray(points, dtype=float)
    points = np.unique(points, axis=0)
    if len(points) < min_points or len(points) < 3:
        return AlphaShapeResult(triangles=np.empty((0, 3, 2)), area_m2=0.0)

    try:
        delaunay = Delaunay(points)
    except Exception:
        return AlphaShapeResult(triangles=np.empty((0, 3, 2)), area_m2=0.0)

    kept: list[np.ndarray] = []
    area = 0.0
    max_radius = 1.0 / max(float(alpha), 1e-9)
    for simplex in delaunay.simplices:
        tri = points[simplex]
        radius = _circumradius(tri)
        if np.isfinite(radius) and radius <= max_radius:
            kept.append(tri)
            area += _triangle_area(tri)
    if not kept:
        return AlphaShapeResult(triangles=np.empty((0, 3, 2)), area_m2=0.0)
    return AlphaShapeResult(triangles=np.asarray(kept), area_m2=float(area))


def summarize_regions(clustered_points: pd.DataFrame, alpha: float, min_points: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if clustered_points.empty:
        return pd.DataFrame(columns=["destination_label", "point_count", "alpha_area_m2", "centroid_x", "centroid_y"])
    for label, group in clustered_points.groupby("destination_label", sort=True):
        coords = group[["x", "y"]].to_numpy(dtype=float)
        shape = alpha_shape(coords, alpha=alpha, min_points=min_points)
        rows.append(
            {
                "destination_label": label,
                "point_count": len(group),
                "alpha_area_m2": shape.area_m2,
                "centroid_x": float(np.mean(coords[:, 0])),
                "centroid_y": float(np.mean(coords[:, 1])),
            }
        )
    return pd.DataFrame(rows)


def summarize_cluster_alpha_shapes(
    clustered_points: pd.DataFrame, alpha: float, min_points: int
) -> pd.DataFrame:
    columns = [
        "destination_label",
        "selected_cluster_id",
        "point_count",
        "alpha",
        "triangle_count",
        "alpha_area_m2",
        "centroid_x",
        "centroid_y",
        "noise_point_count",
    ]
    if clustered_points.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for label, label_points in clustered_points.groupby("destination_label", sort=True):
        non_noise = label_points[label_points["cluster_id"] != "noise"]
        noise_count = int((label_points["cluster_id"] == "noise").sum())
        if non_noise.empty:
            selected = label_points
            cluster_id = "all_points_no_cluster"
        else:
            cluster_counts = non_noise.groupby("cluster_id").size().sort_values(ascending=False)
            cluster_id = str(cluster_counts.index[0])
            selected = non_noise[non_noise["cluster_id"] == cluster_id]

        coords = selected[["x", "y"]].to_numpy(dtype=float)
        shape = alpha_shape(coords, alpha=alpha, min_points=min_points)
        centroid = coords.mean(axis=0)
        rows.append(
            {
                "destination_label": label,
                "selected_cluster_id": cluster_id,
                "point_count": len(selected),
                "alpha": float(alpha),
                "triangle_count": int(len(shape.triangles)),
                "alpha_area_m2": float(shape.area_m2),
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "noise_point_count": noise_count,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def correct_trajectories_to_table_targets(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    table_points: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[tuple[str, str, pd.DataFrame]], pd.DataFrame]:
    target_radius_m = float(config.get("target_radius_m", 0.45))
    progress_power = float(config.get("progress_power", 1.0))
    targets = _table_point_targets(table_points)
    corrected_trajectories: list[tuple[str, str, pd.DataFrame]] = []
    summary_rows: list[dict[str, Any]] = []

    for trial_id, label, trajectory in trajectories:
        if trajectory.empty or label not in targets:
            corrected_trajectories.append((trial_id, label, trajectory.copy()))
            continue

        target = targets[label]
        endpoint = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        offset_from_target = endpoint - target
        endpoint_distance = float(np.linalg.norm(offset_from_target))
        if endpoint_distance > target_radius_m and endpoint_distance > 1e-9:
            corrected_endpoint = target + offset_from_target / endpoint_distance * target_radius_m
        else:
            corrected_endpoint = endpoint.copy()

        correction = corrected_endpoint - endpoint
        corrected = trajectory.copy()
        progress = _trajectory_progress(len(corrected), progress_power)
        corrected["x"] = corrected["x"].to_numpy(dtype=float) + progress * correction[0]
        corrected["y"] = corrected["y"].to_numpy(dtype=float) + progress * correction[1]
        corrected["correction_dx_m"] = progress * correction[0]
        corrected["correction_dy_m"] = progress * correction[1]
        corrected["correction_progress"] = progress
        corrected_trajectories.append((trial_id, label, corrected))

        corrected_endpoint_distance = float(np.linalg.norm(corrected_endpoint - target))
        summary_rows.append(
            {
                "trial_id": trial_id,
                "destination_label": label,
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_radius_m": target_radius_m,
                "original_endpoint_x": float(endpoint[0]),
                "original_endpoint_y": float(endpoint[1]),
                "corrected_endpoint_x": float(corrected_endpoint[0]),
                "corrected_endpoint_y": float(corrected_endpoint[1]),
                "endpoint_distance_to_target_before_m": endpoint_distance,
                "endpoint_distance_to_target_after_m": corrected_endpoint_distance,
                "endpoint_correction_dx_m": float(correction[0]),
                "endpoint_correction_dy_m": float(correction[1]),
                "endpoint_correction_distance_m": float(np.linalg.norm(correction)),
                "progress_power": progress_power,
            }
        )

    return corrected_trajectories, pd.DataFrame(summary_rows)


def recompute_segment_centers_from_trajectories(
    segments: pd.DataFrame,
    trajectories: list[tuple[str, str, pd.DataFrame]],
) -> pd.DataFrame:
    if segments.empty:
        return segments.copy()

    corrected = segments.copy()
    trajectory_by_trial = {trial_id: trajectory for trial_id, _, trajectory in trajectories}
    for index, segment in corrected.iterrows():
        trajectory = trajectory_by_trial.get(str(segment["trial_id"]))
        if trajectory is None or trajectory.empty:
            continue
        start = int(segment["start_row"])
        end = int(segment["end_row"])
        selected = trajectory.iloc[start:end]
        if selected.empty:
            continue
        corrected.at[index, "segment_center_x"] = float(selected["x"].mean())
        corrected.at[index, "segment_center_y"] = float(selected["y"].mean())
        corrected.at[index, "representative_x"] = float(selected["x"].mean())
        corrected.at[index, "representative_y"] = float(selected["y"].mean())
    return corrected


def classify_table_area_segments(segments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return _classify_segments(segments, config)


def _segment_features(
    trial_id: str,
    destination_label: str,
    segment_id: int,
    start: int,
    end: int,
    segment: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    x = segment["x"].to_numpy(dtype=float)
    y = segment["y"].to_numpy(dtype=float)
    speed = segment["speed_mps"].to_numpy(dtype=float)
    length = _polyline_length(x, y)
    displacement = float(np.hypot(x[-1] - x[0], y[-1] - y[0])) if len(segment) > 1 else 0.0
    tortuosity_ratio = length / max(displacement, 1e-9) if displacement > 0 else 1.0
    duration = float(segment["step_time"].iloc[-1] - segment["step_time"].iloc[0]) if len(segment) > 1 else 0.0
    heading_change = float(segment["heading_delta_rad"].sum())
    cumulative_heading_change_deg = float(np.degrees(heading_change))
    progress_end = float(segment["progress_ratio"].iloc[-1])
    slowdown_ratio = float(segment["is_slowdown"].mean())
    low_speed_ratio = float(segment["is_low_speed"].mean())
    speed_cv = float(np.std(speed) / max(np.mean(speed), 1e-9))
    selected_table_points = _select_table_area_points(segment)
    segment_center_x = float(segment["x"].mean())
    segment_center_y = float(segment["y"].mean())

    max_table_segment_length = float(config.get("max_curve_segment_length_m", 1.2))
    speed_table_evidence = (
        slowdown_ratio >= float(config.get("min_slowdown_ratio", 0.20))
        and length <= max_table_segment_length
    )
    max_turn_evidence = float(segment["heading_delta_deg"].max()) >= float(
        config.get("curve_segment_turn_deg", 25.0)
    )
    cumulative_curve_evidence = cumulative_heading_change_deg >= float(
        config.get("cumulative_curve_turn_deg", 60.0)
    )
    nearby_curve_evidence = cumulative_heading_change_deg >= float(
        config.get("nearby_curve_turn_deg", 45.0)
    )
    tortuosity_curve_evidence = tortuosity_ratio >= float(
        config.get("curve_tortuosity_ratio", 1.20)
    )
    curve_evidence = (
        max_turn_evidence
        or cumulative_curve_evidence
        or tortuosity_curve_evidence
    )

    return {
        "trial_id": trial_id,
        "destination_label": destination_label,
        "segment_id": segment_id,
        "start_row": start,
        "end_row": end,
        "step_count": len(segment),
        "start_step_index": int(segment["step_index"].iloc[0]),
        "end_step_index": int(segment["step_index"].iloc[-1]),
        "start_time_s": float(segment["step_time"].iloc[0]),
        "end_time_s": float(segment["step_time"].iloc[-1]),
        "duration_s": duration,
        "path_length_m": length,
        "displacement_m": displacement,
        "tortuosity_ratio": tortuosity_ratio,
        "mean_speed_mps": float(np.mean(speed)),
        "speed_cv": speed_cv,
        "max_speed_change_ratio": float(segment["speed_change_ratio"].max()),
        "max_speed_drop_ratio": float(segment["speed_drop_ratio"].max()),
        "slowdown_ratio": slowdown_ratio,
        "low_speed_ratio": low_speed_ratio,
        "heading_change_deg": cumulative_heading_change_deg,
        "max_heading_delta_deg": float(segment["heading_delta_deg"].max()),
        "has_max_turn_evidence": max_turn_evidence,
        "has_cumulative_curve_evidence": cumulative_curve_evidence,
        "has_tortuosity_curve_evidence": tortuosity_curve_evidence,
        "progress_start": float(segment["progress_ratio"].iloc[0]),
        "progress_end": progress_end,
        "distance_to_endpoint_m": float(segment["distance_to_endpoint_m"].iloc[-1]),
        "slowdown_followed_by_turn": bool(segment["slowdown_followed_by_turn"].any()),
        "has_speed_table_evidence": speed_table_evidence,
        "has_curve_evidence": curve_evidence,
        "has_nearby_curve_evidence": nearby_curve_evidence,
        "is_near_speed_table_segment": False,
        "has_slowdown_evidence": speed_table_evidence,
        "classification": "corridor",
        "representative_x": float(selected_table_points["x"].mean()),
        "representative_y": float(selected_table_points["y"].mean()),
        "segment_center_x": segment_center_x,
        "segment_center_y": segment_center_y,
    }


def _classify_segments(segments: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    if segments.empty:
        return segments

    classified = segments.copy()
    classified["is_near_speed_table_segment"] = False
    speed_table = classified["has_speed_table_evidence"].to_numpy(dtype=bool)
    curve_heading = classified["heading_change_deg"].to_numpy(dtype=float) >= float(
        config.get("table_curve_heading_deg", 45.0)
    )
    short_curve = classified["path_length_m"].to_numpy(dtype=float) <= float(
        config.get("max_curve_segment_length_m", 1.2)
    )
    near_distance = float(config.get("nearby_curve_distance_m", 1.2))
    curve_near_segment_gap = int(config.get("curve_near_segment_gap", 1))

    for i, row in classified.iterrows():
        if speed_table[int(i)] or not (curve_heading[int(i)] and short_curve[int(i)]):
            continue
        for _, seed in classified[classified["has_speed_table_evidence"]].iterrows():
            spatially_near = _segment_distance_m(row, seed) <= near_distance
            same_trace = (
                str(row["trial_id"]) == str(seed["trial_id"])
                and str(row["destination_label"]) == str(seed["destination_label"])
            )
            sequentially_near = same_trace and abs(int(row["segment_id"]) - int(seed["segment_id"])) <= curve_near_segment_gap
            if spatially_near or sequentially_near:
                classified.at[i, "is_near_speed_table_segment"] = True
                break

    table_area = (
        classified["has_speed_table_evidence"]
        | classified["is_near_speed_table_segment"]
    )
    classified["classification"] = np.where(table_area, "seat_area", "corridor")
    classified["has_slowdown_evidence"] = classified["has_speed_table_evidence"]
    return classified


def _segment_distance_m(a: pd.Series, b: pd.Series) -> float:
    ax = float(a["representative_x"])
    ay = float(a["representative_y"])
    bx = float(b["representative_x"])
    by = float(b["representative_y"])
    return float(np.hypot(ax - bx, ay - by))


def _polyline_length(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.hypot(np.diff(x), np.diff(y)).sum())


def _representative_point_rows(segment_row: dict[str, Any], segment: pd.DataFrame) -> list[dict[str, Any]]:
    if segment_row["classification"] != "seat_area":
        return []
    selected = _select_table_area_points(segment)
    rows = []
    for _, point in selected.iterrows():
        rows.append(
            {
                "trial_id": segment_row["trial_id"],
                "destination_label": segment_row["destination_label"],
                "segment_id": segment_row["segment_id"],
                "step_index": int(point["step_index"]),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "speed_mps": float(point["speed_mps"]),
                "speed_drop_ratio": float(point["speed_drop_ratio"]),
                "speed_change_ratio": float(point["speed_change_ratio"]),
                "is_low_speed": bool(point["is_low_speed"]),
                "is_slowdown": bool(point["is_slowdown"]),
                "progress_ratio": float(point["progress_ratio"]),
            }
        )
    return rows


def select_table_area_points_for_segment(segment: pd.DataFrame) -> pd.DataFrame:
    return _select_table_area_points(segment)


def _select_table_area_points(segment: pd.DataFrame) -> pd.DataFrame:
    mask = (
        segment["is_slowdown"].to_numpy(dtype=bool)
        | segment["is_low_speed"].to_numpy(dtype=bool)
    )
    selected = segment[mask]
    return selected if not selected.empty else segment.tail(1)


def _merge_short_intervals(intervals: list[tuple[int, int]], min_steps: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if merged and end - start < min_steps:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_steps:
        first_start, _ = merged.pop(0)
        next_start, next_end = merged[0]
        merged[0] = (min(first_start, next_start), next_end)
    return merged


def _previous_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        start = max(0, i - window)
        previous = values[start:i]
        result[i] = float(np.mean(previous)) if len(previous) else float(values[i])
    return result


def _future_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        end = min(len(values), i + max(window, 1))
        future = values[i:end]
        result[i] = float(np.mean(future)) if len(future) else float(values[i])
    return result


def _table_point_targets(table_points: pd.DataFrame) -> dict[str, np.ndarray]:
    targets: dict[str, np.ndarray] = {}
    if table_points.empty:
        return targets
    for _, row in table_points.iterrows():
        targets[str(row["destination_label"])] = np.array(
            [float(row["centroid_x"]), float(row["centroid_y"])],
            dtype=float,
        )
    return targets


def _trajectory_progress(length: int, power: float) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 0), dtype=float)
    base = np.linspace(0.0, 1.0, length)
    return np.power(base, max(power, 1e-9))


def _slowdown_followed_by_turn(
    slowdown: np.ndarray, heading_delta: np.ndarray, window: int, turn_threshold: float
) -> np.ndarray:
    result = np.zeros(len(slowdown), dtype=bool)
    for i, is_slow in enumerate(slowdown):
        if not is_slow:
            continue
        end = min(len(slowdown), i + window + 1)
        result[i:end] |= np.any(heading_delta[i:end] >= turn_threshold)
    return result


def _dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    labels = np.full(len(points), -1, dtype=int)
    visited = np.zeros(len(points), dtype=bool)
    cluster_id = 0
    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = _region_query(points, i, eps)
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        seeds = list(neighbors)
        j = 0
        while j < len(seeds):
            point_index = seeds[j]
            if not visited[point_index]:
                visited[point_index] = True
                point_neighbors = _region_query(points, point_index, eps)
                if len(point_neighbors) >= min_samples:
                    seeds.extend(n for n in point_neighbors if n not in seeds)
            if labels[point_index] == -1:
                labels[point_index] = cluster_id
            j += 1
        cluster_id += 1
    return labels


def _region_query(points: np.ndarray, index: int, eps: float) -> list[int]:
    distances = np.linalg.norm(points - points[index], axis=1)
    return np.flatnonzero(distances <= eps).tolist()


def _angle_differences(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (target - reference)))


def _circumradius(triangle: np.ndarray) -> float:
    a = np.linalg.norm(triangle[1] - triangle[0])
    b = np.linalg.norm(triangle[2] - triangle[1])
    c = np.linalg.norm(triangle[0] - triangle[2])
    area = _triangle_area(triangle)
    if area <= 1e-12:
        return float("inf")
    return float(a * b * c / (4.0 * area))


def _triangle_area(triangle: np.ndarray) -> float:
    return float(abs(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) * 0.5)


def _empty_segments() -> pd.DataFrame:
    return pd.DataFrame()


def _empty_points() -> pd.DataFrame:
    return pd.DataFrame(columns=["trial_id", "destination_label", "segment_id", "step_index", "x", "y"])
