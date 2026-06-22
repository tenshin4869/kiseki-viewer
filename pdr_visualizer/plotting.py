from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib
from typing import Any

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import pandas as pd

from pdr_visualizer.table_region import alpha_shape

plt.rcParams["font.family"] = [
    "Hiragino Sans",
    "Noto Sans JP",
    "AppleGothic",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def plot_acc_norm(acc_df: pd.DataFrame, steps_df: pd.DataFrame, output_path: str | Path, dpi: int, show_grid: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acc_df["t"], acc_df["step_signal_raw"], label="歩行信号（生値）", alpha=0.45)
    ax.plot(acc_df["t"], acc_df["step_signal_smooth"], label="歩行信号（平滑化）")
    if not steps_df.empty:
        ax.scatter(steps_df["step_time"], steps_df["step_signal_smooth"], label="検出歩行ステップ", s=20, zorder=3)
    ax.set_xlabel("時間 [秒]")
    ax.set_ylabel("歩行信号 [m/s^2]")
    ax.legend()
    ax.grid(show_grid)
    _save(fig, output_path, dpi)


def plot_heading(heading_df: pd.DataFrame, gyro_axis: str, output_path: str | Path, dpi: int, show_grid: bool) -> None:
    axis_col = f"gyro_{gyro_axis}"
    corrected_col = f"{axis_col}_corrected"
    if "gyro_vertical_corrected" in heading_df.columns:
        axis_col = "gyro_vertical_raw"
        corrected_col = "gyro_vertical_corrected"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(heading_df["t"], heading_df[axis_col], label=axis_col, alpha=0.5)
    axes[0].plot(heading_df["t"], heading_df[corrected_col], label=corrected_col)
    axes[0].set_ylabel("角速度 [rad/s]")
    axes[0].legend()
    axes[0].grid(show_grid)
    axes[1].plot(heading_df["t"], heading_df["heading_rad"], label="方位角")
    axes[1].set_xlabel("時間 [秒]")
    axes[1].set_ylabel("方位角 [rad]")
    axes[1].legend()
    axes[1].grid(show_grid)
    _save(fig, output_path, dpi)


def plot_magnetic_diagnostics(
    heading_df: pd.DataFrame, output_path: str | Path, dpi: int, show_grid: bool
) -> None:
    if "mag_norm_ut" not in heading_df.columns:
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(heading_df["t"], heading_df["mag_norm_ut"], label="磁場強度")
    axes[0].axhline(
        float(heading_df["mag_norm_ut"].median()),
        color="tab:gray",
        linestyle="--",
        label="中央値",
    )
    axes[0].set_ylabel("磁場 [uT]")
    axes[0].legend()
    axes[1].plot(
        heading_df["t"],
        np.degrees(heading_df["mag_heading_error_rad"]),
        label="磁気方位との差",
    )
    axes[1].plot(
        heading_df["t"],
        np.degrees(heading_df["mag_correction_applied_rad"]),
        label="適用補正量",
    )
    axes[1].set_ylabel("角度 [度]")
    axes[1].legend()
    axes[2].plot(
        heading_df["t"],
        heading_df["mag_validation_accepted"].astype(int),
        label="磁気補正採用",
    )
    axes[2].set_ylabel("採用")
    axes[2].set_xlabel("時間 [秒]")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend()
    for ax in axes:
        ax.grid(show_grid)
    _save(fig, output_path, dpi)


def plot_trajectory(
    trajectory_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_trajectory(ax, trajectory_df, label=title)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    _save(fig, output_path, dpi)


def plot_trajectory_comparison(
    corrected_df: pd.DataFrame,
    simple_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for label, trajectory_df, color in (
        ("磁気・ジャイロ融合", corrected_df, "tab:blue"),
        ("シンプル版", simple_df, "tab:orange"),
    ):
        x = [0.0, *trajectory_df["x"].to_list()]
        y = [0.0, *trajectory_df["y"].to_list()]
        ax.plot(x, y, marker="o", markersize=3, label=label, color=color)
        if len(x) > 1:
            ax.scatter([x[-1]], [y[-1]], marker="x", s=55, color=color, zorder=4)
    ax.scatter([0.0], [0.0], marker="s", s=50, color="tab:green", label="開始地点", zorder=4)
    ax.set_title(f"{title} 軌跡比較")
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend()
    _save(fig, output_path, dpi)


def plot_overlay(
    trajectories: list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for trial_id, trajectory_df in trajectories:
        _draw_trajectory(ax, trajectory_df, label=trial_id, mark_start=False, mark_end=False)
    ax.scatter([0.0], [0.0], marker="s", s=50, color="tab:green", label="開始地点", zorder=4)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    legend_columns = 2 if len(trajectories) > 12 else 1
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small", ncols=legend_columns)
    _save(fig, output_path, dpi)


def plot_table_regions(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    points_df: pd.DataFrame,
    table_points_df: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = {"A": "tab:blue", "C": "tab:orange"}
    _draw_table_overlays(ax, table_overlays or {}, colors)
    _draw_background_trajectories(ax, trajectories, colors)
    _draw_seat_area_segment_paths(ax, trajectories, segments_df, colors)

    if not points_df.empty:
        for label, group in points_df.groupby("destination_label", sort=True):
            color = colors.get(label, "tab:gray")
            ax.scatter(
                group["x"],
                group["y"],
                s=22,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                label=f"{label} 席周辺セグメント点",
            )

    if not table_points_df.empty:
        for _, row in table_points_df.iterrows():
            label = str(row["destination_label"])
            color = colors.get(label, "tab:gray")
            ax.scatter(
                [float(row["centroid_x"])],
                [float(row["centroid_y"])],
                marker="*",
                s=260,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                label=f"{label} 推定テーブル代表点",
                zorder=25,
            )
            ax.text(
                float(row["centroid_x"]) + 0.10,
                float(row["centroid_y"]) + 0.10,
                f"{label} 推定点",
                color=color,
                fontsize=10,
                weight="bold",
                zorder=26,
            )

    ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="入口", zorder=5)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def plot_table_region_classification_clean(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    table_points_df: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = {"A": "tab:blue", "C": "tab:orange"}
    _draw_table_overlays(ax, table_overlays or {}, colors)

    trajectory_by_trial = {trial_id: trajectory for trial_id, _, trajectory in trajectories}
    labeled: set[str] = set()
    for trial_id, label, trajectory in trajectories:
        if trajectory.empty:
            continue
        color = colors.get(label, "tab:gray")
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            color=color,
            alpha=0.10,
            linewidth=1.0,
            label=f"{label} 元軌跡" if label not in labeled else None,
            zorder=3,
        )
        labeled.add(label)

    if not segments_df.empty:
        for _, segment in segments_df.sort_values(["trial_id", "segment_id"]).iterrows():
            trajectory = trajectory_by_trial.get(str(segment["trial_id"]))
            if trajectory is None:
                continue
            start = int(segment["start_row"])
            end = int(segment["end_row"])
            draw_end = min(end + 1, len(trajectory))
            selected = trajectory.iloc[start:draw_end]
            if selected.empty:
                continue
            color = colors.get(str(segment["destination_label"]), "tab:gray")
            is_seat_area = str(segment["classification"]) == "seat_area"
            ax.plot(
                selected["x"],
                selected["y"],
                color=color,
                alpha=0.84 if is_seat_area else 0.13,
                linewidth=3.0 if is_seat_area else 1.0,
                marker="o" if is_seat_area else None,
                markersize=3,
                solid_capstyle="round",
                zorder=10 if is_seat_area else 4,
            )
        _draw_segment_boundary_cuts(ax, trajectory_by_trial, segments_df)

    if not table_points_df.empty:
        for _, row in table_points_df.iterrows():
            label = str(row["destination_label"])
            color = colors.get(label, "tab:gray")
            ax.scatter(
                [float(row["centroid_x"])],
                [float(row["centroid_y"])],
                marker="*",
                s=260,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                label=f"{label} 推定点",
                zorder=30,
            )
    ax.scatter([0.0], [0.0], marker="s", s=70, color="tab:green", label="入口", zorder=31)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def plot_table_region_ellipses(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    clustered_segment_points_df: pd.DataFrame,
    ellipse_df: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = {"A": "tab:blue", "C": "tab:orange"}
    _draw_table_overlays(ax, table_overlays or {}, colors)

    if not clustered_segment_points_df.empty:
        for label, group in clustered_segment_points_df.groupby("destination_label", sort=True):
            color = colors.get(str(label), "tab:gray")
            selected = group[group["cluster_id"] != "noise"]
            if not selected.empty:
                cluster_id = selected.groupby("cluster_id").size().sort_values(ascending=False).index[0]
                selected = selected[selected["cluster_id"] == cluster_id]
            else:
                selected = group
            ax.scatter(
                selected["x"],
                selected["y"],
                s=46,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{label} セグメント中心点",
                zorder=24,
            )

    if not ellipse_df.empty:
        for _, row in ellipse_df.iterrows():
            label = str(row["destination_label"])
            color = colors.get(label, "tab:gray")
            center_x = float(row["center_x"])
            center_y = float(row["center_y"])
            width = 2.0 * float(row["semi_major_m"])
            height = 2.0 * float(row["semi_minor_m"])
            ellipse = Ellipse(
                (center_x, center_y),
                width=width,
                height=height,
                angle=float(row["angle_deg"]),
                facecolor=color,
                edgecolor=color,
                linewidth=2.0,
                alpha=0.18,
                zorder=18,
            )
            ax.add_patch(ellipse)
            ax.scatter(
                [center_x],
                [center_y],
                marker="*",
                s=260,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                label=f"{label} 楕円中心",
                zorder=30,
            )
            ax.text(
                center_x + 0.12,
                center_y + 0.12,
                f"{label} 楕円中心",
                color=color,
                fontsize=10,
                weight="bold",
                zorder=31,
            )

    ax.scatter([0.0], [0.0], marker="s", s=70, color="tab:green", label="入口", zorder=32)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def plot_table_region_alpha_shapes(
    clustered_segment_points_df: pd.DataFrame,
    alpha: float,
    min_points: int,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = {"A": "tab:blue", "C": "tab:orange"}
    _draw_table_overlays(ax, table_overlays or {}, colors)

    if not clustered_segment_points_df.empty:
        for label, group in clustered_segment_points_df.groupby("destination_label", sort=True):
            color = colors.get(str(label), "tab:gray")
            selected = group[group["cluster_id"] != "noise"]
            if not selected.empty:
                cluster_id = selected.groupby("cluster_id").size().sort_values(ascending=False).index[0]
                selected = selected[selected["cluster_id"] == cluster_id]
            else:
                selected = group

            coords = selected[["x", "y"]].to_numpy(dtype=float)
            shape = alpha_shape(coords, alpha=alpha, min_points=min_points)
            if len(shape.triangles):
                collection = PolyCollection(
                    shape.triangles,
                    facecolors=color,
                    edgecolors=color,
                    linewidths=1.5,
                    alpha=0.20,
                    zorder=16,
                )
                ax.add_collection(collection)

            ax.scatter(
                selected["x"],
                selected["y"],
                s=48,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                label=f"{label} セグメント中心点",
                zorder=24,
            )
            center = coords.mean(axis=0)
            ax.scatter(
                [float(center[0])],
                [float(center[1])],
                marker="*",
                s=260,
                color=color,
                edgecolor="black",
                linewidth=1.0,
                label=f"{label} α-shape中心",
                zorder=30,
            )
            ax.text(
                float(center[0]) + 0.12,
                float(center[1]) + 0.12,
                f"{label} 中心",
                color=color,
                fontsize=10,
                weight="bold",
                zorder=31,
            )

    ax.scatter([0.0], [0.0], marker="s", s=70, color="tab:green", label="入口", zorder=32)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def _draw_background_trajectories(
    ax: plt.Axes,
    trajectories: list[tuple[str, str, pd.DataFrame]],
    colors: dict[str, str],
) -> None:
    labeled: set[str] = set()
    for _, label, trajectory in trajectories:
        if trajectory.empty:
            continue
        color = colors.get(label, "tab:gray")
        legend_label = f"{label} 元の軌跡" if label not in labeled else None
        labeled.add(label)
        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color=color,
            alpha=0.12,
            linewidth=1.0,
            label=legend_label,
            zorder=1,
        )


def _draw_seat_area_segment_paths(
    ax: plt.Axes,
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    colors: dict[str, str],
) -> None:
    if segments_df.empty:
        return
    trajectory_by_id = {trial_id: trajectory for trial_id, _, trajectory in trajectories}
    labeled: set[str] = set()
    seat_segments = segments_df[segments_df["classification"] == "seat_area"]
    for _, segment in seat_segments.iterrows():
        trial_id = str(segment["trial_id"])
        label = str(segment["destination_label"])
        trajectory = trajectory_by_id.get(trial_id)
        if trajectory is None:
            continue
        start = int(segment["start_row"])
        end = int(segment["end_row"])
        draw_end = min(end + 1, len(trajectory))
        selected = trajectory.iloc[start:draw_end]
        if selected.empty:
            continue
        color = colors.get(label, "tab:gray")
        legend_label = f"{label} 席周辺セグメント軌跡" if label not in labeled else None
        labeled.add(label)
        if len(selected) == 1:
            ax.scatter(
                selected["x"],
                selected["y"],
                s=26,
                color=color,
                alpha=0.28,
                label=legend_label,
                zorder=2,
            )
        else:
            ax.plot(
                selected["x"],
                selected["y"],
                color=color,
                alpha=0.58,
                linewidth=1.9,
                marker="o",
                markersize=3.8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.35,
                solid_capstyle="round",
                label=legend_label,
                zorder=2,
            )


def _draw_segment_boundary_cuts(
    ax: plt.Axes,
    trajectory_by_trial: dict[str, pd.DataFrame],
    segments_df: pd.DataFrame,
) -> None:
    for _, segment in segments_df.sort_values(["trial_id", "segment_id"]).iterrows():
        if int(segment["segment_id"]) == 0:
            continue
        trajectory = trajectory_by_trial.get(str(segment["trial_id"]))
        if trajectory is None:
            continue
        idx = int(segment["start_row"])
        if idx <= 0 or idx >= len(trajectory):
            continue
        point = trajectory.iloc[idx]
        previous = trajectory.iloc[idx - 1]
        next_point = trajectory.iloc[idx + 1] if idx + 1 < len(trajectory) else point
        dx = float(next_point["x"] - previous["x"])
        dy = float(next_point["y"] - previous["y"])
        norm = float(np.hypot(dx, dy))
        if norm < 1e-9:
            continue
        nx = -dy / norm
        ny = dx / norm
        half = 0.09
        ax.plot(
            [float(point["x"]) - nx * half, float(point["x"]) + nx * half],
            [float(point["y"]) - ny * half, float(point["y"]) + ny * half],
            color="black",
            linewidth=1.0,
            alpha=0.38,
            zorder=18,
        )


def plot_segmentation_overview(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    speed_features_df: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_table_overlays(ax, table_overlays or {}, {"A": "tab:blue", "C": "tab:orange"})
    trajectory_by_trial = {trial_id: trajectory for trial_id, _, trajectory in trajectories}
    _draw_classified_segments(ax, trajectory_by_trial, segments_df)
    _draw_slowdown_points(ax, speed_features_df)
    ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="入口", zorder=30)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def plot_all_trajectories_segmentation_plane(
    trajectories: list[tuple[str, str, pd.DataFrame]],
    segments_df: pd.DataFrame,
    speed_features_df: pd.DataFrame,
    table_points_df: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_table_overlays(ax, table_overlays or {}, {"A": "tab:blue", "C": "tab:orange"})

    trajectory_by_trial = {trial_id: trajectory for trial_id, _, trajectory in trajectories}
    labels_seen: set[str] = set()
    for trial_id, destination_label, trajectory in trajectories:
        if trajectory.empty:
            continue
        color = "tab:blue" if destination_label == "A" else "tab:orange"
        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color=color,
            linewidth=0.9,
            alpha=0.18,
            label=f"{destination_label} 元の軌跡" if f"{destination_label} 元の軌跡" not in labels_seen else None,
            zorder=3,
        )
        labels_seen.add(f"{destination_label} 元の軌跡")

    _draw_classified_segments(ax, trajectory_by_trial, segments_df)
    _draw_slowdown_points(ax, speed_features_df)
    _draw_clustered_table_points(ax, table_points_df, {"A": "tab:blue", "C": "tab:orange"})
    ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="入口", zorder=30)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    _save(fig, output_path, dpi)


def plot_trial_segmentation_diagnostic(
    trial_id: str,
    trajectory_df: pd.DataFrame,
    trial_segments: pd.DataFrame,
    trial_speed_features: pd.DataFrame,
    table_overlays: dict[str, dict[str, Any]] | None,
    output_path: str | Path,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 16), height_ratios=[1.05, 1.05, 0.95])
    ax_split, ax_classification, ax_metrics = axes

    _draw_table_overlays(ax_split, table_overlays or {}, {"A": "tab:blue", "C": "tab:orange"})
    _draw_segment_split_map(ax_split, trajectory_df, trial_segments)
    ax_split.scatter([0.0], [0.0], marker="s", s=55, color="tab:green", label="入口", zorder=30)
    segment_count = len(trial_segments)
    ax_split.set_title(f"{trial_id}: どこで軌跡を分割したか（全{segment_count}セグメント）")
    _style_trajectory_axis(ax_split, equal_axis, show_grid)
    ax_split.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")

    _draw_table_overlays(ax_classification, table_overlays or {}, {"A": "tab:blue", "C": "tab:orange"})
    _draw_classified_segments(ax_classification, {trial_id: trajectory_df}, trial_segments)
    _draw_slowdown_points(ax_classification, trial_speed_features)
    ax_classification.scatter([0.0], [0.0], marker="s", s=55, color="tab:green", label="入口", zorder=30)
    seat_count = int((trial_segments["classification"] == "seat_area").sum()) if not trial_segments.empty else 0
    corridor_count = segment_count - seat_count
    ax_classification.set_title(
        f"{trial_id}: 分割後の通路/席周辺分類（通路{corridor_count}・テーブル周辺{seat_count}）"
    )
    _style_trajectory_axis(ax_classification, equal_axis, show_grid)
    ax_classification.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")

    x = trial_speed_features["step_index"].to_numpy(dtype=float)
    ax_metrics.plot(
        x,
        trial_speed_features["speed_mps"],
        marker="o",
        color="tab:blue",
        label="歩行速度 [m/s]",
    )
    ax_metrics.plot(
        x,
        trial_speed_features["previous_speed_mps"],
        color="tab:cyan",
        linestyle="--",
        label="直前平均速度 [m/s]",
    )
    ax_drop = ax_metrics.twinx()
    ax_drop.plot(
        x,
        trial_speed_features["speed_drop_ratio"],
        marker=".",
        color="tab:red",
        label="速度低下率",
    )
    if "speed_change_ratio" in trial_speed_features.columns:
        ax_drop.plot(
            x,
            trial_speed_features["speed_change_ratio"],
            marker=".",
            color="tab:orange",
            alpha=0.75,
            label="速度変化率",
        )
    ax_drop.plot(
        x,
        trial_speed_features["heading_delta_deg"] / 100.0,
        color="tab:purple",
        alpha=0.65,
        label="方向変化 / 100",
    )
    slowdown = trial_speed_features[trial_speed_features["is_slowdown"]]
    if not slowdown.empty:
        ax_metrics.scatter(
            slowdown["step_index"],
            slowdown["speed_mps"],
            marker="v",
            s=75,
            color="gold",
            edgecolor="black",
            linewidth=0.5,
            label="減速点",
            zorder=10,
        )
    for _, segment in trial_segments.iterrows():
        start = int(segment["start_step_index"])
        end = int(segment["end_step_index"])
        color = "#f2b8b5" if segment["classification"] == "seat_area" else "#d9d9d9"
        ax_metrics.axvspan(start - 0.5, end + 0.5, color=color, alpha=0.35, linewidth=0)
        ax_metrics.axvline(start - 0.5, color="0.25", linewidth=0.7, alpha=0.55)
    ax_metrics.set_xlabel("ステップ番号")
    ax_metrics.set_ylabel("歩行速度 [m/s]")
    ax_drop.set_ylabel("比率 / スケール済み角度")
    ax_metrics.grid(show_grid)
    lines, labels = ax_metrics.get_legend_handles_labels()
    drop_lines, drop_labels = ax_drop.get_legend_handles_labels()
    ax_metrics.legend(lines + drop_lines, labels + drop_labels, loc="upper left", fontsize="small")
    _save(fig, output_path, dpi)


def _draw_table_overlays(
    ax: plt.Axes, table_overlays: dict[str, dict[str, Any]], colors: dict[str, str]
) -> None:
    for label, table in table_overlays.items():
        color = "black"
        left = float(table["left"])
        bottom = float(table["bottom"])
        width = float(table["width"])
        height = float(table["height"])
        ax.add_patch(
            Rectangle(
                (left, bottom),
                width,
                height,
                facecolor="white",
                edgecolor=color,
                linewidth=1.8,
                linestyle="--",
                alpha=0.70,
                zorder=0,
            )
        )
        ax.text(
            left + width / 2,
            bottom + height / 2,
            str(table.get("label", f"Table {label}")),
            color=color,
            fontsize=9,
            ha="center",
            va="center",
            weight="bold",
            alpha=0.70,
            zorder=0.5,
        )


def _draw_clustered_table_points(
    ax: plt.Axes, table_points_df: pd.DataFrame, colors: dict[str, str]
) -> None:
    if table_points_df.empty:
        return
    for _, row in table_points_df.iterrows():
        label = str(row["destination_label"])
        color = colors.get(label, "tab:gray")
        x = float(row["centroid_x"])
        y = float(row["centroid_y"])
        ax.scatter(
            [x],
            [y],
            marker="*",
            s=320,
            color=color,
            edgecolor="black",
            linewidth=1.1,
            label=f"{label} クラスタ推定点",
            zorder=35,
        )
        ax.text(
            x + 0.12,
            y + 0.12,
            f"{label} 推定点",
            color=color,
            fontsize=10,
            weight="bold",
            zorder=36,
        )


def _draw_segment_split_map(ax: plt.Axes, trajectory_df: pd.DataFrame, segments_df: pd.DataFrame) -> None:
    if trajectory_df.empty:
        return

    ax.plot(
        trajectory_df["x"],
        trajectory_df["y"],
        color="0.75",
        linewidth=1.0,
        alpha=0.8,
        label="元の軌跡",
        zorder=4,
    )
    ax.scatter(
        trajectory_df["x"],
        trajectory_df["y"],
        s=18,
        color="white",
        edgecolor="0.35",
        linewidth=0.7,
        label="歩行ステップ",
        zorder=9,
    )

    if not segments_df.empty:
        cmap = plt.get_cmap("tab20")
        for _, segment in segments_df.iterrows():
            start = int(segment["start_row"])
            end = int(segment["end_row"])
            draw_end = min(end + 1, len(trajectory_df))
            selected = trajectory_df.iloc[start:draw_end]
            if selected.empty:
                continue
            segment_id = int(segment["segment_id"])
            color = cmap(segment_id % 20)
            if len(selected) == 1:
                ax.scatter(
                    selected["x"],
                    selected["y"],
                    s=90,
                    color=color,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=13,
                )
            else:
                ax.plot(
                    selected["x"],
                    selected["y"],
                    color=color,
                    linewidth=3.0,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=12,
                )
            mid = selected.iloc[len(selected) // 2]
            ax.text(
                float(mid["x"]),
                float(mid["y"]),
                str(segment_id),
                ha="center",
                va="center",
                fontsize=9,
                weight="bold",
                color="black",
                bbox={"boxstyle": "circle,pad=0.25", "facecolor": "white", "edgecolor": color, "linewidth": 1.2},
                zorder=25,
            )
            ax.scatter(
                [selected["x"].iloc[0]],
                [selected["y"].iloc[0]],
                marker="|",
                s=130,
                color="black",
                alpha=0.8,
                label="セグメント境界" if segment_id == 0 else None,
                zorder=22,
            )

    ax.scatter(
        [trajectory_df["x"].iloc[0]],
        [trajectory_df["y"].iloc[0]],
        marker="o",
        s=55,
        color="tab:green",
        edgecolor="black",
        linewidth=0.5,
        label="軌跡開始",
        zorder=28,
    )
    ax.scatter(
        [trajectory_df["x"].iloc[-1]],
        [trajectory_df["y"].iloc[-1]],
        marker="X",
        s=65,
        color="tab:red",
        edgecolor="black",
        linewidth=0.5,
        label="軌跡終了",
        zorder=28,
    )


def _draw_classified_segments(
    ax: plt.Axes, trajectory_by_trial: dict[str, pd.DataFrame], segments_df: pd.DataFrame
) -> None:
    if segments_df.empty:
        return
    labels_seen: set[str] = set()

    for trial_id in segments_df["trial_id"].drop_duplicates():
        trajectory = trajectory_by_trial.get(str(trial_id))
        if trajectory is None or trajectory.empty:
            continue
        ax.plot(
            trajectory["x"],
            trajectory["y"],
            color="0.15",
            linewidth=0.8,
            alpha=0.25,
            label="元の軌跡" if "元の軌跡" not in labels_seen else None,
            zorder=4,
        )
        labels_seen.add("元の軌跡")

    for _, segment in segments_df.iterrows():
        trial_id = str(segment["trial_id"])
        trajectory = trajectory_by_trial.get(trial_id)
        if trajectory is None:
            continue
        start = int(segment["start_row"])
        end = int(segment["end_row"])
        draw_end = min(end + 1, len(trajectory))
        selected = trajectory.iloc[start:draw_end]
        if selected.empty:
            continue
        classification = str(segment["classification"])
        if classification == "seat_area":
            color = "tab:red"
            linewidth = 3.0
            alpha = 0.9
            label = "席周辺セグメント"
            zorder = 12
        else:
            color = "0.65"
            linewidth = 1.6
            alpha = 0.35
            label = "通路セグメント"
            zorder = 8
        if len(selected) == 1:
            ax.scatter(
                selected["x"],
                selected["y"],
                s=90 if classification == "seat_area" else 45,
                color=color,
                alpha=alpha,
                edgecolor="black" if classification == "seat_area" else "none",
                linewidth=0.6,
                label=label if label not in labels_seen else None,
                zorder=zorder,
            )
        else:
            ax.plot(
                selected["x"],
                selected["y"],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                label=label if label not in labels_seen else None,
                zorder=zorder,
                solid_capstyle="round",
            )
        labels_seen.add(label)
        if classification == "seat_area":
            mid = selected.iloc[len(selected) // 2]
            ax.text(
                float(mid["x"]) + 0.05,
                float(mid["y"]) + 0.05,
                "テーブル周辺",
                color="tab:red",
                fontsize=8,
                weight="bold",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "tab:red",
                    "linewidth": 0.8,
                    "alpha": 0.85,
                },
                zorder=26,
            )
        ax.scatter(
            [selected["x"].iloc[0]],
            [selected["y"].iloc[0]],
            marker="|",
            s=95,
            color="black",
            alpha=0.65,
            label="セグメント境界" if "セグメント境界" not in labels_seen else None,
            zorder=20,
        )
        labels_seen.add("セグメント境界")


def _draw_slowdown_points(ax: plt.Axes, speed_features_df: pd.DataFrame) -> None:
    if speed_features_df.empty or "is_slowdown" not in speed_features_df.columns:
        return
    slowdown = speed_features_df[speed_features_df["is_slowdown"]]
    if slowdown.empty:
        return
    ax.scatter(
        slowdown["x"],
        slowdown["y"],
        marker="v",
        s=70,
        color="gold",
        edgecolor="black",
        linewidth=0.55,
        label="減速点",
        zorder=25,
    )


def _draw_trajectory(
    ax: plt.Axes,
    trajectory_df: pd.DataFrame,
    label: str,
    mark_start: bool = True,
    mark_end: bool = True,
) -> None:
    x = [0.0, *trajectory_df["x"].to_list()]
    y = [0.0, *trajectory_df["y"].to_list()]
    ax.plot(x, y, marker="o", markersize=3, label=label)
    if mark_start:
        ax.scatter([0.0], [0.0], marker="s", s=50, color="tab:green", label="開始地点", zorder=4)
    if mark_end and len(x) > 1:
        ax.scatter([x[-1]], [y[-1]], marker="x", s=55, color="tab:red", label="終了地点", zorder=4)


def _style_trajectory_axis(ax: plt.Axes, equal_axis: bool, show_grid: bool) -> None:
    ax.set_xlabel("x座標 [m]")
    ax.set_ylabel("y座標 [m]")
    if equal_axis:
        ax.axis("equal")
    ax.grid(show_grid)


def _save(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
