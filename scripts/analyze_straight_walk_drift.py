from __future__ import annotations

import argparse
import math
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze step-by-step and meter-by-meter drift in 10 m straight-walk PDR trials."
    )
    parser.add_argument(
        "--manifest",
        default="output/a_straight_walk_error_corrected/correction_trials/manifest.csv",
    )
    parser.add_argument("--output-dir", default="output/straight_walk_drift_analysis")
    parser.add_argument("--true-distance-m", type=float, default=10.0)
    args = parser.parse_args()

    if args.true_distance_m <= 0:
        raise ValueError("--true-distance-m must be positive")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest.get("status", "processed") == "processed"].copy()
    trial_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    meter_rows: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        trajectory = pd.read_csv(row["trajectory_csv"])
        if trajectory.empty:
            continue
        trial_summary, trial_steps, trial_meters = _analyze_trial(
            row,
            trajectory,
            args.true_distance_m,
        )
        trial_rows.append(trial_summary)
        step_rows.extend(trial_steps)
        meter_rows.extend(trial_meters)

    trial_summary = pd.DataFrame(trial_rows)
    step_samples = pd.DataFrame(step_rows)
    meter_samples = pd.DataFrame(meter_rows)
    meter_profile = _summarize_meter_profile(meter_samples)
    group_summary = _summarize_groups(trial_summary)
    overall_summary = _summarize_overall(trial_summary)

    trial_summary.to_csv(output_dir / "trial_drift_summary.csv", index=False)
    step_samples.to_csv(output_dir / "step_drift_samples.csv", index=False)
    meter_samples.to_csv(output_dir / "meter_drift_samples.csv", index=False)
    meter_profile.to_csv(output_dir / "meter_drift_profile.csv", index=False)
    group_summary.to_csv(output_dir / "group_drift_summary.csv", index=False)
    overall_summary.to_csv(output_dir / "overall_drift_summary.csv", index=False)

    _plot_lateral_by_meter(meter_profile, figures_dir / "abs_lateral_drift_by_meter.png")
    _plot_angle_by_meter(meter_profile, figures_dir / "abs_bearing_drift_by_meter.png")
    _plot_endpoint_histogram(trial_summary, figures_dir / "endpoint_lateral_drift_histogram.png")
    _plot_group_boxplot(trial_summary, figures_dir / "endpoint_lateral_drift_by_group.png")

    print(f"Trials analyzed: {len(trial_summary)}")
    print(f"Step samples: {len(step_samples)}")
    print(f"Output dir: {output_dir}")
    if not overall_summary.empty:
        row = overall_summary.iloc[0]
        print(
            "Overall: "
            f"abs_endpoint_x_median={row['abs_endpoint_x_m_median']:.6f} m, "
            f"abs_endpoint_x_per_true_m_median={row['abs_endpoint_x_per_true_m_median']:.6f} m/m, "
            f"abs_endpoint_bearing_deg_median={row['abs_endpoint_bearing_deg_median']:.6f} deg, "
            f"abs_endpoint_bearing_deg_per_true_m_median="
            f"{row['abs_endpoint_bearing_deg_per_true_m_median']:.6f} deg/m"
        )


def _analyze_trial(
    manifest_row: pd.Series,
    trajectory: pd.DataFrame,
    true_distance_m: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    x = trajectory["x"].to_numpy(dtype=float)
    y = trajectory["y"].to_numpy(dtype=float)
    headings = np.unwrap(trajectory["heading_rad"].to_numpy(dtype=float))
    step_lengths = trajectory["step_length_m"].to_numpy(dtype=float)
    n_steps = len(trajectory)
    cumulative_path = np.cumsum(step_lengths)
    pdr_path_length = float(cumulative_path[-1])
    normalized_distance = cumulative_path / pdr_path_length * true_distance_m
    endpoint_x = float(x[-1])
    endpoint_y = float(y[-1])
    endpoint_bearing_rad = math.atan2(endpoint_x, endpoint_y)
    endpoint_bearing_deg = math.degrees(endpoint_bearing_rad)
    heading_change_deg = math.degrees(float(headings[-1] - headings[0])) if n_steps > 1 else 0.0

    prev_x = np.concatenate([[0.0], x[:-1]])
    prev_y = np.concatenate([[0.0], y[:-1]])
    dx = x - prev_x
    dy = y - prev_y
    local_heading_deg = np.degrees(np.unwrap(np.arctan2(dx, dy)))
    cumulative_bearing_deg = np.degrees(np.arctan2(x, y))
    heading_deg = np.degrees(headings)
    heading_drift_from_initial_deg = heading_deg - heading_deg[0]
    delta_heading_deg = np.concatenate([[0.0], np.diff(heading_deg)])

    step_rows: list[dict[str, Any]] = []
    base = {
        "group": manifest_row.get("group"),
        "trial_id": manifest_row.get("trial_id"),
        "source_path": manifest_row.get("source_path"),
    }
    for i in range(n_steps):
        per_true_meter = x[i] / normalized_distance[i] if normalized_distance[i] > 0 else np.nan
        per_pdr_meter = x[i] / cumulative_path[i] if cumulative_path[i] > 0 else np.nan
        step_rows.append(
            {
                **base,
                "step_index": int(trajectory["step_index"].iloc[i]),
                "step_number": i + 1,
                "step_length_m": float(step_lengths[i]),
                "cumulative_pdr_distance_m": float(cumulative_path[i]),
                "normalized_true_distance_m": float(normalized_distance[i]),
                "x_m": float(x[i]),
                "y_m": float(y[i]),
                "abs_x_m": float(abs(x[i])),
                "signed_lateral_drift_per_true_m": float(per_true_meter),
                "abs_lateral_drift_per_true_m": float(abs(per_true_meter)),
                "signed_lateral_drift_per_pdr_m": float(per_pdr_meter),
                "abs_lateral_drift_per_pdr_m": float(abs(per_pdr_meter)),
                "cumulative_bearing_deg": float(cumulative_bearing_deg[i]),
                "abs_cumulative_bearing_deg": float(abs(cumulative_bearing_deg[i])),
                "local_heading_deg": float(local_heading_deg[i]),
                "abs_local_heading_deg": float(abs(local_heading_deg[i])),
                "heading_deg": float(heading_deg[i]),
                "heading_drift_from_initial_deg": float(heading_drift_from_initial_deg[i]),
                "abs_heading_drift_from_initial_deg": float(abs(heading_drift_from_initial_deg[i])),
                "delta_heading_deg": float(delta_heading_deg[i]),
                "abs_delta_heading_deg": float(abs(delta_heading_deg[i])),
            }
        )

    meter_rows: list[dict[str, Any]] = []
    for meter in np.arange(1.0, true_distance_m + 0.001, 1.0):
        interp_x = float(np.interp(meter, normalized_distance, x))
        interp_y = float(np.interp(meter, normalized_distance, y))
        interp_heading = float(np.interp(meter, normalized_distance, heading_deg))
        interp_heading_drift = interp_heading - heading_deg[0]
        bearing_deg = math.degrees(math.atan2(interp_x, interp_y))
        meter_rows.append(
            {
                **base,
                "meter": int(round(meter)),
                "normalized_true_distance_m": float(meter),
                "x_m": interp_x,
                "y_m": interp_y,
                "abs_x_m": abs(interp_x),
                "signed_lateral_drift_per_true_m": interp_x / meter,
                "abs_lateral_drift_per_true_m": abs(interp_x) / meter,
                "cumulative_bearing_deg": bearing_deg,
                "abs_cumulative_bearing_deg": abs(bearing_deg),
                "heading_deg": interp_heading,
                "heading_drift_from_initial_deg": interp_heading_drift,
                "abs_heading_drift_from_initial_deg": abs(interp_heading_drift),
            }
        )

    trial_summary = {
        **base,
        "step_count": n_steps,
        "pdr_path_length_m": pdr_path_length,
        "endpoint_x_m": endpoint_x,
        "endpoint_y_m": endpoint_y,
        "abs_endpoint_x_m": abs(endpoint_x),
        "endpoint_x_per_step_m": endpoint_x / n_steps,
        "abs_endpoint_x_per_step_m": abs(endpoint_x) / n_steps,
        "endpoint_x_per_true_m": endpoint_x / true_distance_m,
        "abs_endpoint_x_per_true_m": abs(endpoint_x) / true_distance_m,
        "endpoint_x_per_pdr_m": endpoint_x / pdr_path_length,
        "abs_endpoint_x_per_pdr_m": abs(endpoint_x) / pdr_path_length,
        "endpoint_bearing_deg": endpoint_bearing_deg,
        "abs_endpoint_bearing_deg": abs(endpoint_bearing_deg),
        "endpoint_bearing_deg_per_step": endpoint_bearing_deg / n_steps,
        "abs_endpoint_bearing_deg_per_step": abs(endpoint_bearing_deg) / n_steps,
        "endpoint_bearing_deg_per_true_m": endpoint_bearing_deg / true_distance_m,
        "abs_endpoint_bearing_deg_per_true_m": abs(endpoint_bearing_deg) / true_distance_m,
        "heading_change_deg": heading_change_deg,
        "abs_heading_change_deg": abs(heading_change_deg),
        "heading_change_deg_per_step": heading_change_deg / max(n_steps - 1, 1),
        "abs_heading_change_deg_per_step": abs(heading_change_deg) / max(n_steps - 1, 1),
        "heading_change_deg_per_true_m": heading_change_deg / true_distance_m,
        "abs_heading_change_deg_per_true_m": abs(heading_change_deg) / true_distance_m,
        "mean_abs_local_heading_deg": float(np.mean(np.abs(local_heading_deg))),
        "median_abs_local_heading_deg": float(np.median(np.abs(local_heading_deg))),
        "mean_abs_delta_heading_deg": float(np.mean(np.abs(delta_heading_deg[1:])))
        if n_steps > 1
        else 0.0,
        "median_abs_delta_heading_deg": float(np.median(np.abs(delta_heading_deg[1:])))
        if n_steps > 1
        else 0.0,
    }
    return trial_summary, step_rows, meter_rows


def _summarize_meter_profile(meter_samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for meter, group in meter_samples.groupby("meter"):
        rows.append(
            {
                "meter": int(meter),
                "sample_count": int(len(group)),
                **_series_stats(group["x_m"], "signed_x_m"),
                **_series_stats(group["abs_x_m"], "abs_x_m"),
                **_series_stats(group["abs_lateral_drift_per_true_m"], "abs_x_per_true_m"),
                **_series_stats(group["abs_cumulative_bearing_deg"], "abs_bearing_deg"),
                **_series_stats(
                    group["abs_heading_drift_from_initial_deg"],
                    "abs_heading_drift_from_initial_deg",
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_groups(trial_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in trial_summary.groupby("group"):
        rows.append(
            {
                "group": group_name,
                "sample_count": int(len(group)),
                **_series_stats(group["endpoint_x_m"], "signed_endpoint_x_m"),
                **_series_stats(group["abs_endpoint_x_m"], "abs_endpoint_x_m"),
                **_series_stats(group["abs_endpoint_x_per_true_m"], "abs_endpoint_x_per_true_m"),
                **_series_stats(group["abs_endpoint_bearing_deg"], "abs_endpoint_bearing_deg"),
                **_series_stats(
                    group["abs_endpoint_bearing_deg_per_true_m"],
                    "abs_endpoint_bearing_deg_per_true_m",
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_overall(trial_summary: pd.DataFrame) -> pd.DataFrame:
    if trial_summary.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "sample_count": int(len(trial_summary)),
                **_series_stats(trial_summary["endpoint_x_m"], "signed_endpoint_x_m"),
                **_series_stats(trial_summary["abs_endpoint_x_m"], "abs_endpoint_x_m"),
                **_series_stats(
                    trial_summary["abs_endpoint_x_per_step_m"],
                    "abs_endpoint_x_per_step_m",
                ),
                **_series_stats(
                    trial_summary["abs_endpoint_x_per_true_m"],
                    "abs_endpoint_x_per_true_m",
                ),
                **_series_stats(
                    trial_summary["abs_endpoint_x_per_pdr_m"],
                    "abs_endpoint_x_per_pdr_m",
                ),
                **_series_stats(
                    trial_summary["abs_endpoint_bearing_deg"],
                    "abs_endpoint_bearing_deg",
                ),
                **_series_stats(
                    trial_summary["abs_endpoint_bearing_deg_per_step"],
                    "abs_endpoint_bearing_deg_per_step",
                ),
                **_series_stats(
                    trial_summary["abs_endpoint_bearing_deg_per_true_m"],
                    "abs_endpoint_bearing_deg_per_true_m",
                ),
                **_series_stats(
                    trial_summary["abs_heading_change_deg"],
                    "abs_heading_change_deg",
                ),
                **_series_stats(
                    trial_summary["abs_heading_change_deg_per_step"],
                    "abs_heading_change_deg_per_step",
                ),
                **_series_stats(
                    trial_summary["abs_heading_change_deg_per_true_m"],
                    "abs_heading_change_deg_per_true_m",
                ),
            }
        ]
    )


def _series_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q90": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=1)),
        f"{prefix}_q25": float(values.quantile(0.25)),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q75": float(values.quantile(0.75)),
        f"{prefix}_q90": float(values.quantile(0.90)),
        f"{prefix}_max": float(values.max()),
    }


def _plot_lateral_by_meter(profile: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(profile["meter"], profile["abs_x_m_median"], marker="o", label="median")
    ax.fill_between(
        profile["meter"],
        profile["abs_x_m_q25"],
        profile["abs_x_m_q75"],
        alpha=0.25,
        label="IQR",
    )
    ax.plot(profile["meter"], profile["abs_x_m_q90"], linestyle="--", label="90th percentile")
    ax.set_title("Absolute lateral drift by normalized walking distance")
    ax.set_xlabel("normalized walking distance [m]")
    ax.set_ylabel("absolute lateral drift |x| [m]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_angle_by_meter(profile: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(profile["meter"], profile["abs_bearing_deg_median"], marker="o", label="median")
    ax.fill_between(
        profile["meter"],
        profile["abs_bearing_deg_q25"],
        profile["abs_bearing_deg_q75"],
        alpha=0.25,
        label="IQR",
    )
    ax.plot(profile["meter"], profile["abs_bearing_deg_q90"], linestyle="--", label="90th percentile")
    ax.set_title("Absolute cumulative bearing drift by normalized walking distance")
    ax.set_xlabel("normalized walking distance [m]")
    ax.set_ylabel("absolute cumulative bearing drift [deg]")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_endpoint_histogram(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(summary["abs_endpoint_x_m"], bins=14, color="tab:blue", alpha=0.75)
    ax.axvline(summary["abs_endpoint_x_m"].median(), color="black", linestyle="--", label="median")
    ax.set_title("Endpoint absolute lateral drift distribution")
    ax.set_xlabel("absolute endpoint lateral drift |x| [m]")
    ax.set_ylabel("trial count")
    ax.grid(True, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_group_boxplot(summary: pd.DataFrame, output_path: Path) -> None:
    groups = [group["abs_endpoint_x_m"].to_numpy() for _, group in summary.groupby("group")]
    labels = [str(name) for name, _ in summary.groupby("group")]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(groups, tick_labels=labels, showmeans=True)
    ax.set_title("Endpoint absolute lateral drift by correction group")
    ax.set_xlabel("group")
    ax.set_ylabel("absolute endpoint lateral drift |x| [m]")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
