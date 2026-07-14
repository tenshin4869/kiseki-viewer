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
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.table_region import classify_table_area_segments, segment_trajectory


SEAT_ORDER = ("A-1", "A-2", "A-3", "A-4")
SEAT_COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}
TARGETS = {
    "A-1": (-2.0, 4.6),
    "A-2": (-2.7, 4.6),
    "A-3": (-2.0, 5.5),
    "A-4": (-2.7, 5.5),
}
TABLE_A = {
    "left": -3.3,
    "bottom": 4.6,
    "width": 1.5,
    "height": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment A-table trajectories and plot seat-area segments.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_segments")
    parser.add_argument(
        "--direct-slowdown-only",
        action="store_true",
        help="Classify only direct slowdown evidence as seat_area; do not include nearby curve segments.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    segmentation_config = config["table_region"]["segmentation"]
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["trajectory_name"].isin(SEAT_ORDER)].copy()

    trajectories: list[dict[str, Any]] = []
    all_segments = []
    all_points = []
    for _, row in manifest.iterrows():
        trial_id = str(row["trial_id"])
        seat = str(row["trajectory_name"])
        trajectory = pd.read_csv(row["trajectory_csv"])
        segments, points = segment_trajectory(trial_id, seat, trajectory, segmentation_config)
        all_segments.append(segments)
        all_points.append(points)
        trajectories.append(
            {
                "trial_id": trial_id,
                "dataset": str(row["dataset"]),
                "seat": seat,
                "trajectory": trajectory,
            }
        )

    segments_df = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    segments_df = classify_table_area_segments(segments_df, segmentation_config)
    if args.direct_slowdown_only and not segments_df.empty:
        segments_df["is_near_speed_table_segment"] = False
        segments_df["classification"] = segments_df["has_speed_table_evidence"].map(
            {True: "seat_area", False: "corridor"}
        )
    points_df = pd.concat(all_points, ignore_index=True) if all_points else pd.DataFrame()
    summary = _summarize_segments(segments_df)

    segments_path = output_dir / "segments.csv"
    points_path = output_dir / "segment_points.csv"
    summary_path = output_dir / "segment_summary.csv"
    segments_df.to_csv(segments_path, index=False)
    points_df.to_csv(points_path, index=False)
    summary.to_csv(summary_path, index=False)

    _plot_all(trajectories, segments_df, figures_dir / "a_segments_overall.png")
    _plot_grid(trajectories, segments_df, figures_dir / "a_segments_by_seat_2x2.png")

    print(f"Segments: {segments_path}")
    print(f"Segment points: {points_path}")
    print(f"Summary: {summary_path}")
    print(f"Overall figure: {figures_dir / 'a_segments_overall.png'}")
    print(f"2x2 figure: {figures_dir / 'a_segments_by_seat_2x2.png'}")
    print(summary.to_string(index=False))


def _summarize_segments(segments: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    rows = []
    for seat, group in segments.groupby("destination_label", sort=True):
        rows.append(
            {
                "seat": seat,
                "trial_count": int(group["trial_id"].nunique()),
                "segment_count": int(len(group)),
                "corridor_segment_count": int((group["classification"] == "corridor").sum()),
                "seat_area_segment_count": int((group["classification"] == "seat_area").sum()),
                "mean_segments_per_trial": float(len(group) / group["trial_id"].nunique()),
                "mean_seat_area_segments_per_trial": float(
                    (group["classification"] == "seat_area").sum() / group["trial_id"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_all(trajectories: list[dict[str, Any]], segments: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_panel(ax, trajectories, segments, title="A-seat trajectory segmentation overview")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(trajectories: list[dict[str, Any]], segments: pd.DataFrame, output_path: Path) -> None:
    limits = _limits(trajectories)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, seat in zip(axes.ravel(), SEAT_ORDER):
        seat_trajectories = [row for row in trajectories if row["seat"] == seat]
        seat_segments = segments[segments["destination_label"] == seat]
        _draw_panel(ax, seat_trajectories, seat_segments, title=seat, include_legend=False)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    handles = [
        plt.Line2D([0], [0], color="0.70", linewidth=1.0, label="trajectory / corridor"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="seat-area segment"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="true seat"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:green", markersize=8, label="start"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=5, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    fig.suptitle("A-seat segmentation by seat", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_panel(
    ax: plt.Axes,
    trajectories: list[dict[str, Any]],
    segments: pd.DataFrame,
    *,
    title: str,
    include_legend: bool = True,
) -> None:
    by_trial = {row["trial_id"]: row for row in trajectories}
    labeled: set[str] = set()
    for row in trajectories:
        seat = row["seat"]
        trajectory = row["trajectory"]
        color = SEAT_COLORS[seat]
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            color=color,
            alpha=0.10,
            linewidth=0.9,
            label=f"{seat} trajectory" if include_legend and seat not in labeled else None,
            zorder=4,
        )
        labeled.add(seat)

    if not segments.empty:
        for _, segment in segments.sort_values(["trial_id", "segment_id"]).iterrows():
            trial_id = str(segment["trial_id"])
            row = by_trial.get(trial_id)
            if row is None:
                continue
            trajectory = row["trajectory"]
            start = int(segment["start_row"])
            end = min(int(segment["end_row"]) + 1, len(trajectory))
            selected = trajectory.iloc[start:end]
            if selected.empty:
                continue
            seat = str(segment["destination_label"])
            is_seat_area = str(segment["classification"]) == "seat_area"
            ax.plot(
                selected["x"],
                selected["y"],
                color=SEAT_COLORS[seat] if is_seat_area else "0.62",
                alpha=0.82 if is_seat_area else 0.18,
                linewidth=3.2 if is_seat_area else 1.0,
                marker="o" if is_seat_area else None,
                markersize=3.0,
                solid_capstyle="round",
                zorder=8 if is_seat_area else 4,
            )

    _draw_targets(ax)
    _draw_table_a(ax)
    ax.scatter([0.0], [0.0], marker="s", s=58, color="tab:green", label="start" if include_legend else None, zorder=10)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _draw_targets(ax: plt.Axes) -> None:
    for seat, (x, y) in TARGETS.items():
        ax.scatter([x], [y], marker="*", s=210, color="gold", edgecolor="black", linewidth=0.9, zorder=12)
        ax.annotate(seat, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)


def _draw_table_a(ax: plt.Axes) -> None:
    rectangle = plt.Rectangle(
        (TABLE_A["left"], TABLE_A["bottom"]),
        TABLE_A["width"],
        TABLE_A["height"],
        fill=False,
        edgecolor="black",
        linewidth=3.0,
        zorder=3,
    )
    ax.add_patch(rectangle)


def _limits(trajectories: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = [0.0]
    ys = [0.0]
    for row in trajectories:
        xs.extend(row["trajectory"]["x"].to_list())
        ys.extend(row["trajectory"]["y"].to_list())
    for x, y in TARGETS.values():
        xs.append(x)
        ys.append(y)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.5, (x_max - x_min) * 0.08)
    pad_y = max(0.5, (y_max - y_min) * 0.08)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


if __name__ == "__main__":
    main()
