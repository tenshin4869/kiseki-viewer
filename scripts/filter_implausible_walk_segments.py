from __future__ import annotations

import argparse
import os
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


COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}
SEAT_ORDER = ("A-1", "A-2", "A-3", "A-4")
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
    parser = argparse.ArgumentParser(
        description="Trim implausible post-stop walking tails without using map or ground-truth seats."
    )
    parser.add_argument("--manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_trajectories_first_stop_filtered")
    parser.add_argument("--stop-gap-s", type=float, default=3.0)
    parser.add_argument("--min-steps-after-trim", type=int, default=4)
    parser.add_argument("--max-step-distance-m", type=float, default=1.2)
    parser.add_argument("--min-step-interval-s", type=float, default=0.25)
    parser.add_argument("--max-step-speed-mps", type=float, default=2.5)
    args = parser.parse_args()

    source_manifest = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    trajectories_dir = output_dir / "trials"
    figures_dir = output_dir / "figures"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    before_endpoints = []
    after_endpoints = []
    before_trajectories: list[dict[str, Any]] = []
    after_trajectories: list[dict[str, Any]] = []

    for _, row in source_manifest.iterrows():
        trajectory = pd.read_csv(row["trajectory_csv"])
        before_trajectories.append(
            {
                "trajectory_name": row["trajectory_name"],
                "trajectory": trajectory,
            }
        )
        metrics = _trajectory_metrics(trajectory)
        trim_index = _first_stop_trim_index(trajectory, args.stop_gap_s)
        trimmed = trajectory.iloc[: trim_index + 1].copy()
        reasons = _removal_reasons(
            metrics,
            len(trimmed),
            min_steps_after_trim=args.min_steps_after_trim,
            max_step_distance_m=args.max_step_distance_m,
            min_step_interval_s=args.min_step_interval_s,
            max_step_speed_mps=args.max_step_speed_mps,
        )

        trial_id = str(row["trial_id"])
        diagnostic_rows.append(
            {
                **row.to_dict(),
                **metrics,
                "trimmed_at_step_index": int(trim_index),
                "trimmed_step_count": int(len(trimmed)),
                "trimmed_endpoint_x": float(trimmed["x"].iloc[-1]) if len(trimmed) else np.nan,
                "trimmed_endpoint_y": float(trimmed["y"].iloc[-1]) if len(trimmed) else np.nan,
                "was_trimmed": bool(len(trimmed) != len(trajectory)),
                "removed": bool(reasons),
                "remove_reason": "; ".join(reasons),
            }
        )

        before_endpoints.append(
            {
                "trajectory_name": row["trajectory_name"],
                "x": row["endpoint_x"],
                "y": row["endpoint_y"],
            }
        )

        if reasons:
            removed_rows.append(
                {
                    **row.to_dict(),
                    "trimmed_step_count": int(len(trimmed)),
                    "remove_reason": "; ".join(reasons),
                }
            )
            continue

        trajectory_path = trajectories_dir / trial_id / "trajectory.csv"
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trimmed.to_csv(trajectory_path, index=False)

        kept = row.to_dict()
        kept["step_count"] = int(len(trimmed))
        kept["endpoint_x"] = float(trimmed["x"].iloc[-1])
        kept["endpoint_y"] = float(trimmed["y"].iloc[-1])
        kept["trajectory_csv"] = str(trajectory_path)
        kept["processing_mode"] = f"{row['processing_mode']}+first_stop_gap_trim_{args.stop_gap_s:g}s"
        kept_rows.append(kept)
        after_endpoints.append(
            {
                "trajectory_name": row["trajectory_name"],
                "x": kept["endpoint_x"],
                "y": kept["endpoint_y"],
            }
        )
        after_trajectories.append(
            {
                "trajectory_name": row["trajectory_name"],
                "trajectory": trimmed,
            }
        )

    kept_df = pd.DataFrame(kept_rows)
    removed_df = pd.DataFrame(removed_rows)
    diagnostics_df = pd.DataFrame(diagnostic_rows)
    before_df = pd.DataFrame(before_endpoints)
    after_df = pd.DataFrame(after_endpoints)

    kept_df.to_csv(output_dir / "manifest.csv", index=False)
    removed_df.to_csv(output_dir / "removed_trials.csv", index=False)
    diagnostics_df.to_csv(output_dir / "walk_quality_diagnostics.csv", index=False)
    _plot_endpoint_before_after(
        before_df,
        after_df,
        figures_dir / "endpoint_before_after_first_stop_filter.png",
    )
    _plot_trajectory_overview(
        before_trajectories,
        figures_dir / "a_trajectories_before_first_stop_filter.png",
        "Before: last detected step trajectories",
    )
    _plot_trajectory_overview(
        after_trajectories,
        figures_dir / "a_trajectories_first_stop_filtered_all.png",
        "After: first long-stop trimmed trajectories",
    )
    _plot_trajectory_grid(
        after_trajectories,
        figures_dir / "a_trajectories_first_stop_filtered_by_seat_2x2.png",
    )

    print(f"Input trajectories: {len(source_manifest)}")
    print(f"Kept trajectories: {len(kept_df)}")
    print(f"Removed trajectories: {len(removed_df)}")
    print(f"Trimmed trajectories: {int(diagnostics_df['was_trimmed'].sum())}")
    print(f"Manifest: {output_dir / 'manifest.csv'}")
    print(f"Diagnostics: {output_dir / 'walk_quality_diagnostics.csv'}")
    print(f"Removed: {output_dir / 'removed_trials.csv'}")
    print(f"Figure: {figures_dir / 'endpoint_before_after_first_stop_filter.png'}")
    print(f"Filtered trajectories: {figures_dir / 'a_trajectories_first_stop_filtered_all.png'}")
    print(f"Filtered trajectories 2x2: {figures_dir / 'a_trajectories_first_stop_filtered_by_seat_2x2.png'}")


def _trajectory_metrics(trajectory: pd.DataFrame) -> dict[str, Any]:
    if trajectory.empty:
        return {
            "source_step_count": 0,
            "max_step_distance_m": np.nan,
            "min_step_interval_s": np.nan,
            "max_step_interval_s": np.nan,
            "max_step_speed_mps": np.nan,
            "max_heading_turn_deg": np.nan,
        }

    x = trajectory["x"].to_numpy(float)
    y = trajectory["y"].to_numpy(float)
    times = trajectory["step_time"].to_numpy(float)
    headings = trajectory["heading_rad"].to_numpy(float)
    step_distances = np.hypot(np.diff(np.r_[0.0, x]), np.diff(np.r_[0.0, y]))
    intervals = np.diff(times)
    speeds = step_distances[1:] / intervals if len(intervals) else np.array([])
    turns = np.abs(np.diff(np.unwrap(headings)))
    return {
        "source_step_count": int(len(trajectory)),
        "max_step_distance_m": float(np.nanmax(step_distances)) if len(step_distances) else np.nan,
        "min_step_interval_s": float(np.nanmin(intervals)) if len(intervals) else np.nan,
        "max_step_interval_s": float(np.nanmax(intervals)) if len(intervals) else np.nan,
        "max_step_speed_mps": float(np.nanmax(speeds)) if len(speeds) else np.nan,
        "max_heading_turn_deg": float(np.rad2deg(np.nanmax(turns))) if len(turns) else np.nan,
    }


def _first_stop_trim_index(trajectory: pd.DataFrame, stop_gap_s: float) -> int:
    if len(trajectory) <= 1:
        return max(0, len(trajectory) - 1)
    intervals = np.diff(trajectory["step_time"].to_numpy(float))
    stop_candidates = np.flatnonzero(intervals >= stop_gap_s)
    if len(stop_candidates) == 0:
        return len(trajectory) - 1
    return int(stop_candidates[0])


def _removal_reasons(
    metrics: dict[str, Any],
    trimmed_step_count: int,
    *,
    min_steps_after_trim: int,
    max_step_distance_m: float,
    min_step_interval_s: float,
    max_step_speed_mps: float,
) -> list[str]:
    reasons: list[str] = []
    if trimmed_step_count < min_steps_after_trim:
        reasons.append(f"too few steps after first-stop trim: {trimmed_step_count}")
    if metrics["max_step_distance_m"] > max_step_distance_m:
        reasons.append(f"step distance too large: {metrics['max_step_distance_m']:.2f} m")
    if metrics["min_step_interval_s"] < min_step_interval_s:
        reasons.append(f"step interval too short: {metrics['min_step_interval_s']:.2f} s")
    if metrics["max_step_speed_mps"] > max_step_speed_mps:
        reasons.append(f"step speed too high: {metrics['max_step_speed_mps']:.2f} m/s")
    return reasons


def _plot_endpoint_before_after(before: pd.DataFrame, after: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    for ax, data, title in [
        (axes[0], before, "Before: last detected step endpoint"),
        (axes[1], after, "After: first long-stop endpoint"),
    ]:
        for seat, group in data.groupby("trajectory_name", sort=True):
            ax.scatter(
                group["x"],
                group["y"],
                s=24,
                alpha=0.68,
                color=COLORS.get(str(seat), "tab:gray"),
                edgecolors="none",
                label=str(seat),
                zorder=5,
            )
        _draw_table_a(ax)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        _draw_table_a(ax)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_overview(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_trajectory_panel(ax, rows, title=title, include_legend=True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_trajectory_grid(rows: list[dict[str, Any]], output_path: Path) -> None:
    limits = _trajectory_limits(rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, seat in zip(axes.ravel(), SEAT_ORDER):
        seat_rows = [row for row in rows if str(row["trajectory_name"]) == seat]
        _draw_trajectory_panel(ax, seat_rows, title=seat, include_legend=False)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    handles = [
        plt.Line2D([0], [0], color="0.30", linewidth=1.0, label="trimmed trajectory"),
        plt.Line2D([0], [0], marker="x", color="0.10", markersize=8, linewidth=0, label="trimmed endpoint"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="true seat"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="tab:green", markersize=8, label="start"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=5, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("x [m]")
    fig.supylabel("y [m]")
    fig.suptitle("First long-stop trimmed A trajectories by seat", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_trajectory_panel(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    title: str,
    include_legend: bool,
) -> None:
    labeled: set[str] = set()
    for row in rows:
        seat = str(row["trajectory_name"])
        trajectory = row["trajectory"]
        if trajectory.empty:
            continue
        color = COLORS.get(seat, "tab:gray")
        label = seat if include_legend and seat not in labeled else None
        labeled.add(seat)
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            color=color,
            alpha=0.22,
            linewidth=1.0,
            label=label,
            zorder=6,
        )
        ax.scatter(
            [float(trajectory["x"].iloc[-1])],
            [float(trajectory["y"].iloc[-1])],
            color=color,
            marker="x",
            alpha=0.90,
            s=42,
            linewidths=1.2,
            zorder=8,
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


def _trajectory_limits(rows: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs = [0.0]
    ys = [0.0]
    for row in rows:
        trajectory = row["trajectory"]
        if trajectory.empty:
            continue
        xs.extend(trajectory["x"].to_list())
        ys.extend(trajectory["y"].to_list())
    for x, y in TARGETS.values():
        xs.append(x)
        ys.append(y)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(0.55, (x_max - x_min) * 0.08)
    pad_y = max(0.55, (y_max - y_min) * 0.08)
    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


if __name__ == "__main__":
    main()
