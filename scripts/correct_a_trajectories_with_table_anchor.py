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

from pdr_visualizer.config import load_config


COLORS = {"A-1": "tab:blue", "A-2": "tab:orange", "A-3": "tab:green", "A-4": "tab:red"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Correct each A-table trajectory with one heading-drift rate selected from straight-walk "
            "calibration and a weak table-vicinity anchor."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--correction-summary",
        default="output/straight_walk_drift_analysis/trial_drift_summary.csv",
    )
    parser.add_argument(
        "--trajectory-manifest", default="output/a_trajectories_uncorrected/manifest.csv"
    )
    parser.add_argument("--output-dir", default="output/a_table_anchor_heading_corrected")
    parser.add_argument(
        "--table-margin-m",
        type=float,
        default=0.45,
        help="Allowed distance outside the inferred table rectangle before a geometry penalty starts.",
    )
    parser.add_argument(
        "--grid-step-deg-per-pdr-m", type=float, default=0.01,
        help="Heading-drift candidate spacing. Smaller values are slower but more precise.",
    )
    parser.add_argument(
        "--drift-penalty-weight",
        type=float,
        default=0.25,
        help="Weight for preferring a small heading correction after geometry is satisfied.",
    )
    args = parser.parse_args()
    if args.table_margin_m <= 0 or args.grid_step_deg_per_pdr_m <= 0:
        raise ValueError("--table-margin-m and --grid-step-deg-per-pdr-m must be positive")
    if args.drift_penalty_weight < 0:
        raise ValueError("--drift-penalty-weight must be non-negative")

    config = load_config(args.config)
    table = _table_rectangle(config)
    calibration = pd.read_csv(args.correction_summary)
    heading_model, retained = _heading_prior(calibration)
    source_manifest = pd.read_csv(args.trajectory_manifest, dtype={"dataset": str})

    output_dir = Path(args.output_dir)
    corrected_dir = output_dir / "trials"
    figures_dir = output_dir / "figures"
    corrected_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    candidates = np.arange(
        heading_model["rate_lower_deg_per_pdr_m"],
        heading_model["rate_upper_deg_per_pdr_m"] + args.grid_step_deg_per_pdr_m / 2,
        args.grid_step_deg_per_pdr_m,
    )
    # Zero must be a candidate: an endpoint already in the table vicinity should not move
    # merely because the numerical grid happened not to contain exactly zero.
    candidates = np.unique(np.append(candidates, 0.0))
    correction_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for _, row in source_manifest.iterrows():
        original = pd.read_csv(row["trajectory_csv"])
        if original.empty:
            continue
        result = _select_rate(
            original,
            candidates,
            table,
            args.table_margin_m,
            heading_model["robust_rate_scale_deg_per_pdr_m"],
            args.drift_penalty_weight,
        )
        corrected = _reconstruct(original, result["selected_rate_deg_per_pdr_m"])
        output_path = corrected_dir / Path(str(row["trial_id"]).strip("/")) / "trajectory.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(output_path, index=False)

        before = original[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        after = corrected[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        correction_rows.append(
            {
                "dataset": row.get("dataset"),
                "trajectory_name": row.get("trajectory_name"),
                "trial_id": row.get("trial_id"),
                "original_trajectory_csv": row["trajectory_csv"],
                "corrected_trajectory_csv": str(output_path),
                "original_endpoint_x": before[0],
                "original_endpoint_y": before[1],
                "corrected_endpoint_x": after[0],
                "corrected_endpoint_y": after[1],
                "endpoint_shift_m": float(np.linalg.norm(after - before)),
                **result,
            }
        )
        manifest_row = row.to_dict()
        manifest_row.update(
            {
                "trajectory_csv": str(output_path),
                "endpoint_x": float(after[0]),
                "endpoint_y": float(after[1]),
                "processing_mode": f"{row.get('processing_mode', '')}+table_anchor_heading_correction",
            }
        )
        manifest_rows.append(manifest_row)
        pairs.append({"seat": row["trajectory_name"], "before": original, "after": corrected})

    summary = pd.DataFrame(correction_rows)
    corrected_manifest = pd.DataFrame(manifest_rows)
    retained.to_csv(output_dir / "heading_calibration_retained_trials.csv", index=False)
    pd.DataFrame([heading_model | _table_model(table, args.table_margin_m) | {
        "candidate_grid_step_deg_per_pdr_m": args.grid_step_deg_per_pdr_m,
        "drift_penalty_weight": args.drift_penalty_weight,
        "method": "table_vicinity_anchor_with_distance_accumulated_heading_drift",
    }]).to_csv(output_dir / "correction_model.csv", index=False)
    summary.to_csv(output_dir / "trajectory_correction_summary.csv", index=False)
    corrected_manifest.to_csv(output_dir / "manifest.csv", index=False)
    _plot_before_after(pairs, table, args.table_margin_m, figures_dir / "a_trajectories_table_anchor_before_after.png")
    _plot_endpoint_shifts(summary, table, args.table_margin_m, figures_dir / "a_endpoints_table_anchor_before_after.png")

    changed = int((summary["endpoint_shift_m"] > 1e-9).sum())
    boundary = int(summary["selected_at_rate_bound"].sum())
    print(f"A trajectories corrected: {len(summary)}")
    print(f"Trajectories with non-zero heading correction: {changed}")
    print(f"Selected rate at calibration bound: {boundary}")
    print(
        "Heading prior: "
        f"range=[{heading_model['rate_lower_deg_per_pdr_m']:.3f}, "
        f"{heading_model['rate_upper_deg_per_pdr_m']:.3f}] deg/PDR-m, "
        f"robust scale={heading_model['robust_rate_scale_deg_per_pdr_m']:.3f} deg/PDR-m"
    )
    print(f"Output directory: {output_dir}")


def _table_rectangle(config: dict[str, Any]) -> tuple[float, float, float, float]:
    overlay = config.get("table_region", {}).get("table_overlays", {}).get("A")
    if not overlay:
        raise ValueError("config.table_region.table_overlays.A is required")
    left = float(overlay["left"])
    bottom = float(overlay["bottom"])
    return left, left + float(overlay["width"]), bottom, bottom + float(overlay["height"])


def _heading_prior(summary: pd.DataFrame) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    required = {"abs_endpoint_x_per_true_m", "heading_change_deg", "pdr_path_length_m"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Correction summary is missing columns: {sorted(missing)}")
    valid = summary.dropna(subset=list(required)).copy()
    q1 = float(valid["abs_endpoint_x_per_true_m"].quantile(0.25))
    q3 = float(valid["abs_endpoint_x_per_true_m"].quantile(0.75))
    upper_fence = q3 + 1.5 * (q3 - q1)
    retained = valid[valid["abs_endpoint_x_per_true_m"] <= upper_fence].copy()
    retained["heading_drift_rate_deg_per_pdr_m"] = (
        retained["heading_change_deg"] / retained["pdr_path_length_m"]
    )
    rates = retained["heading_drift_rate_deg_per_pdr_m"].replace([np.inf, -np.inf], np.nan).dropna()
    if rates.empty:
        raise ValueError("No calibration trials remained after IQR filtering")
    rate_q25 = float(rates.quantile(0.25))
    rate_q75 = float(rates.quantile(0.75))
    robust_scale = max((rate_q75 - rate_q25) / 1.349, 1e-6)
    return (
        {
            "calibration_trial_count": int(len(valid)),
            "retained_calibration_trial_count": int(len(rates)),
            "outlier_rule": "abs_endpoint_x_per_true_m > Q3 + 1.5 * IQR",
            "lateral_outlier_upper_fence_m_per_true_m": upper_fence,
            "rate_lower_deg_per_pdr_m": float(rates.quantile(0.05)),
            "rate_upper_deg_per_pdr_m": float(rates.quantile(0.95)),
            "rate_signed_median_deg_per_pdr_m": float(rates.median()),
            "rate_q25_deg_per_pdr_m": rate_q25,
            "rate_q75_deg_per_pdr_m": rate_q75,
            "robust_rate_scale_deg_per_pdr_m": robust_scale,
            "rate_center_for_penalty_deg_per_pdr_m": 0.0,
            "rate_center_reason": "The signed median is near zero; no common left/right systematic drift is assumed.",
        },
        retained,
    )


def _select_rate(
    trajectory: pd.DataFrame,
    candidates: np.ndarray,
    table: tuple[float, float, float, float],
    margin: float,
    robust_rate_scale: float,
    drift_penalty_weight: float,
) -> dict[str, float | bool]:
    endpoints = np.asarray(
        [_reconstruct(trajectory, rate)[["x", "y"]].iloc[-1].to_numpy(dtype=float) for rate in candidates]
    )
    outside_distance = np.asarray([_distance_outside_table_vicinity(point, table, margin) for point in endpoints])
    geometry_loss = (outside_distance / margin) ** 2
    normalized_rate = candidates / robust_rate_scale
    drift_loss = drift_penalty_weight * _huber(normalized_rate, delta=1.5)
    objective = geometry_loss + drift_loss
    selected_index = int(np.argmin(objective))
    original = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
    original_distance = _distance_outside_table_vicinity(original, table, margin)
    return {
        "selected_rate_deg_per_pdr_m": float(candidates[selected_index]),
        "selected_objective": float(objective[selected_index]),
        "original_distance_outside_table_vicinity_m": float(original_distance),
        "corrected_distance_outside_table_vicinity_m": float(outside_distance[selected_index]),
        "geometry_loss": float(geometry_loss[selected_index]),
        "drift_loss": float(drift_loss[selected_index]),
        "selected_at_rate_bound": bool(selected_index in {0, len(candidates) - 1}),
    }


def _reconstruct(trajectory: pd.DataFrame, rate_deg_per_pdr_m: float) -> pd.DataFrame:
    corrected = trajectory.copy()
    headings = trajectory["heading_rad"].to_numpy(dtype=float)
    steps = trajectory["step_length_m"].to_numpy(dtype=float)
    distance = np.cumsum(steps)
    heading_error = np.deg2rad(rate_deg_per_pdr_m * distance)
    corrected["x_original"] = trajectory["x"].to_numpy(dtype=float)
    corrected["y_original"] = trajectory["y"].to_numpy(dtype=float)
    corrected["cumulative_pdr_distance_m"] = distance
    corrected["heading_drift_correction_rad"] = heading_error
    corrected["heading_rad_original"] = headings
    corrected["heading_rad"] = headings + heading_error
    corrected["x"] = np.cumsum(steps * np.sin(corrected["heading_rad"].to_numpy(dtype=float)))
    corrected["y"] = np.cumsum(steps * np.cos(corrected["heading_rad"].to_numpy(dtype=float)))
    corrected["heading_drift_rate_deg_per_pdr_m"] = rate_deg_per_pdr_m
    return corrected


def _distance_outside_table_vicinity(
    point: np.ndarray, table: tuple[float, float, float, float], margin: float
) -> float:
    left, right, bottom, top = table
    x, y = float(point[0]), float(point[1])
    dx = max(left - x, 0.0, x - right)
    dy = max(bottom - y, 0.0, y - top)
    return max(float(np.hypot(dx, dy)) - margin, 0.0)


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute <= delta, 0.5 * values**2, delta * (absolute - 0.5 * delta))


def _table_model(table: tuple[float, float, float, float], margin: float) -> dict[str, float]:
    left, right, bottom, top = table
    return {
        "table_left": left,
        "table_right": right,
        "table_bottom": bottom,
        "table_top": top,
        "table_vicinity_margin_m": margin,
    }


def _plot_before_after(
    pairs: list[dict[str, Any]], table: tuple[float, float, float, float], margin: float, output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ["before", "after"], ["Before correction", "After table-anchor heading correction"]):
        for pair in pairs:
            trajectory = pair[key]
            color = COLORS.get(str(pair["seat"]), "tab:gray")
            ax.plot([0.0, *trajectory["x"]], [0.0, *trajectory["y"]], color=color, alpha=0.17, linewidth=0.8)
            ax.scatter(trajectory["x"].iloc[-1], trajectory["y"].iloc[-1], color=color, s=12, alpha=0.75)
        _draw_table_vicinity(ax, table, margin)
        ax.scatter(0.0, 0.0, marker="s", color="black", s=35)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.grid(True)
    axes[0].set_ylabel("y [m]")
    handles = [plt.Line2D([0], [0], color=color, label=seat) for seat, color in COLORS.items()]
    axes[1].legend(handles=handles, title="seat", loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_shifts(
    summary: pd.DataFrame, table: tuple[float, float, float, float], margin: float, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for seat, group in summary.groupby("trajectory_name"):
        color = COLORS.get(str(seat), "tab:gray")
        ax.scatter(group["original_endpoint_x"], group["original_endpoint_y"], color=color, s=18, alpha=0.28)
        ax.scatter(group["corrected_endpoint_x"], group["corrected_endpoint_y"], color=color, marker="x", s=22, alpha=0.75, label=str(seat))
    _draw_table_vicinity(ax, table, margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Endpoint positions before and after table-anchor correction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend(title="seat")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_table_vicinity(ax: Any, table: tuple[float, float, float, float], margin: float) -> None:
    left, right, bottom, top = table
    ax.add_patch(plt.Rectangle((left, bottom), right - left, top - bottom, fill=False, edgecolor="black", linewidth=1.6, label="table"))
    ax.add_patch(plt.Rectangle((left - margin, bottom - margin), right - left + 2 * margin, top - bottom + 2 * margin, fill=False, edgecolor="black", linewidth=1.0, linestyle="--", alpha=0.7, label="table vicinity"))


if __name__ == "__main__":
    main()
