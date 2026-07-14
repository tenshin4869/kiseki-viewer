from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import zipfile
from copy import deepcopy
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
from pdr_visualizer.trial import run_trial


COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate accumulated PDR error from 10 m straight-walk correction trials "
            "and apply the correction to A-table trajectories."
        )
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--correction-dir", default="data/correction")
    parser.add_argument("--trajectory-manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_straight_walk_error_corrected")
    parser.add_argument("--straight-distance-m", type=float, default=10.0)
    parser.add_argument("--lateral-scale", type=float, default=1.0)
    parser.add_argument(
        "--forward-scale-mode",
        choices=["none", "median"],
        default="none",
        help="Use 'none' to preserve trajectory length, or 'median' to rescale distance using straight-walk trials.",
    )
    parser.add_argument(
        "--lateral-correction-mode",
        choices=["signed_median", "none"],
        default="signed_median",
        help="Use the signed median lateral drift as a deterministic correction, or record uncertainty only.",
    )
    args = parser.parse_args()

    if args.straight_distance_m <= 0:
        raise ValueError("--straight-distance-m must be positive")

    config = load_config(args.config)
    output_root = Path(args.output_dir)
    correction_root = output_root / "correction_trials"
    extracted_root = correction_root / "extracted_raw"
    correction_trials_root = correction_root / "trials"
    corrected_trials_root = output_root / "trials"
    figures_dir = output_root / "figures"
    for path in (extracted_root, correction_trials_root, corrected_trials_root, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    correction_manifest, correction_trajectories = _process_correction_trials(
        Path(args.correction_dir),
        config,
        extracted_root,
        correction_trials_root,
    )
    if correction_manifest.empty:
        raise ValueError(f"No usable correction trials were found in {args.correction_dir}")

    model, model_rows = _estimate_error_model(
        correction_manifest,
        correction_trajectories,
        args.straight_distance_m,
        args.lateral_scale,
        args.forward_scale_mode,
        args.lateral_correction_mode,
    )

    source_manifest = pd.read_csv(args.trajectory_manifest, dtype={"dataset": str})
    corrected_manifest, correction_rows, pairs = _apply_error_model(
        source_manifest,
        model,
        corrected_trials_root,
    )

    correction_manifest_path = correction_root / "manifest.csv"
    model_samples_path = output_root / "straight_walk_error_samples.csv"
    model_summary_path = output_root / "straight_walk_error_model.csv"
    corrected_manifest_path = output_root / "manifest.csv"
    correction_summary_path = output_root / "trajectory_correction_summary.csv"

    correction_manifest.to_csv(correction_manifest_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_samples_path, index=False)
    pd.DataFrame([model]).to_csv(model_summary_path, index=False)
    corrected_manifest.to_csv(corrected_manifest_path, index=False)
    pd.DataFrame(correction_rows).to_csv(correction_summary_path, index=False)

    _plot_straight_walks(
        correction_trajectories,
        args.straight_distance_m,
        figures_dir / "straight_walk_correction_trials.png",
    )
    _plot_overall_pairs(
        pairs,
        config,
        figures_dir / "a_trajectories_before_after_straight_error_correction.png",
    )
    _plot_endpoint_shift(
        pd.DataFrame(correction_rows),
        figures_dir / "a_endpoint_shift_straight_error_correction.png",
    )

    print(f"Correction trials processed: {len(correction_manifest)}")
    print(f"A trajectories corrected: {len(corrected_manifest)}")
    print(
        "Model: "
        f"forward_scale_for_correction={model['forward_scale_for_correction']:.6f}, "
        f"signed_lateral_drift_for_correction={model['lateral_drift_rate_for_correction']:.6f} m/m, "
        f"abs_lateral_endpoint_median={model['abs_endpoint_error_x_m_median']:.6f} m"
    )
    print(f"Correction trial manifest: {correction_manifest_path}")
    print(f"Model samples: {model_samples_path}")
    print(f"Model summary: {model_summary_path}")
    print(f"Corrected manifest: {corrected_manifest_path}")
    print(f"Trajectory correction summary: {correction_summary_path}")
    print(f"Straight-walk figure: {figures_dir / 'straight_walk_correction_trials.png'}")
    print(
        "Before/after figure: "
        f"{figures_dir / 'a_trajectories_before_after_straight_error_correction.png'}"
    )
    print(f"Endpoint-shift figure: {figures_dir / 'a_endpoint_shift_straight_error_correction.png'}")


def _process_correction_trials(
    correction_dir: Path,
    config: dict[str, Any],
    extracted_root: Path,
    trials_output_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    run_config = deepcopy(config)
    run_config["paths"]["raw_data_dir"] = str(extracted_root)
    run_config["paths"]["output_dir"] = str(trials_output_root)

    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    zip_paths = sorted(path for path in correction_dir.rglob("*.zip") if path.is_file())
    for index, zip_path in enumerate(zip_paths, start=1):
        group = zip_path.parent.name
        trial_slug = _trial_slug(zip_path, index)
        extracted_dir = extracted_root / group / trial_slug
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted_dir)

        trial_id = str(extracted_dir.relative_to(extracted_root))
        trial_config = _trial_config_for_available_sensors(run_config, extracted_dir)
        try:
            outputs = run_trial(trial_id, trial_config)
            trajectory = pd.read_csv(outputs["trajectory"])
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "group": group,
                    "source_path": str(zip_path),
                    "trial_id": trial_id,
                    "status": "skipped",
                    "reason": str(exc),
                }
            )
            continue

        rows.append(
            {
                "group": group,
                "source_path": str(zip_path),
                "trial_id": trial_id,
                "status": "processed",
                "step_count": int(len(trajectory)),
                "endpoint_x": float(trajectory["x"].iloc[-1]) if not trajectory.empty else np.nan,
                "endpoint_y": float(trajectory["y"].iloc[-1]) if not trajectory.empty else np.nan,
                "trajectory_csv": str(outputs["trajectory"]),
                "processing_mode": _processing_mode(trial_config),
            }
        )
        trajectories.append(
            {
                "group": group,
                "source_path": str(zip_path),
                "trial_id": trial_id,
                "trajectory": trajectory,
            }
        )

    manifest = pd.DataFrame(rows)
    return manifest[manifest["status"] == "processed"].copy(), trajectories


def _estimate_error_model(
    correction_manifest: pd.DataFrame,
    correction_trajectories: list[dict[str, Any]],
    true_distance_m: float,
    lateral_scale: float,
    forward_scale_mode: str,
    lateral_correction_mode: str,
) -> tuple[dict[str, float | str | int], list[dict[str, float | str | int]]]:
    rows: list[dict[str, float | str | int]] = []
    for item in correction_trajectories:
        trajectory = item["trajectory"]
        if trajectory.empty:
            continue
        endpoint = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        path_length = _trajectory_path_length(trajectory)
        forward_scale = endpoint[1] / true_distance_m
        lateral_drift_rate = endpoint[0] / true_distance_m
        heading_drift_rad_per_m = math.atan2(endpoint[0], endpoint[1]) / true_distance_m
        rows.append(
            {
                "group": item["group"],
                "trial_id": item["trial_id"],
                "source_path": item["source_path"],
                "step_count": int(len(trajectory)),
                "observed_endpoint_x": float(endpoint[0]),
                "observed_endpoint_y": float(endpoint[1]),
                "expected_endpoint_x": 0.0,
                "expected_endpoint_y": float(true_distance_m),
                "endpoint_error_x": float(endpoint[0]),
                "endpoint_error_y": float(endpoint[1] - true_distance_m),
                "observed_path_length_m": path_length,
                "path_length_scale": path_length / true_distance_m,
                "forward_scale": forward_scale,
                "lateral_drift_rate": lateral_drift_rate,
                "heading_drift_deg_per_m": math.degrees(heading_drift_rad_per_m),
            }
        )

    samples = pd.DataFrame(rows)
    valid = samples[
        np.isfinite(samples["forward_scale"])
        & np.isfinite(samples["lateral_drift_rate"])
        & (samples["forward_scale"] > 0)
    ].copy()
    if valid.empty:
        raise ValueError("No valid correction samples were available for model estimation")
    valid["abs_endpoint_error_x"] = valid["endpoint_error_x"].abs()
    valid["abs_lateral_drift_rate"] = valid["lateral_drift_rate"].abs()
    valid["abs_heading_drift_deg_per_m"] = valid["heading_drift_deg_per_m"].abs()

    forward_scale = float(valid["forward_scale"].median())
    forward_scale_for_correction = forward_scale if forward_scale_mode == "median" else 1.0
    lateral_drift_rate = float(valid["lateral_drift_rate"].median()) * lateral_scale
    lateral_drift_rate_for_correction = (
        lateral_drift_rate if lateral_correction_mode == "signed_median" else 0.0
    )
    heading_drift_deg_per_m = float(valid["heading_drift_deg_per_m"].median()) * lateral_scale
    model: dict[str, float | str | int] = {
        "method": "lateral_focused_10m_straight_walk_step_reconstruction",
        "sample_count": int(len(valid)),
        "true_distance_m": float(true_distance_m),
        "forward_scale_mode": forward_scale_mode,
        "lateral_correction_mode": lateral_correction_mode,
        "forward_scale": forward_scale,
        "forward_scale_for_correction": forward_scale_for_correction,
        "lateral_drift_rate_signed_median": lateral_drift_rate,
        "lateral_drift_rate_for_correction": lateral_drift_rate_for_correction,
        "lateral_scale": float(lateral_scale),
        "heading_drift_deg_per_m": heading_drift_deg_per_m,
        "endpoint_error_x_m_median": float(valid["endpoint_error_x"].median()),
        "endpoint_error_y_m_median": float(valid["endpoint_error_y"].median()),
        "endpoint_error_x_m_mean": float(valid["endpoint_error_x"].mean()),
        "endpoint_error_x_m_std": float(valid["endpoint_error_x"].std(ddof=1)),
        "abs_endpoint_error_x_m_mean": float(valid["abs_endpoint_error_x"].mean()),
        "abs_endpoint_error_x_m_std": float(valid["abs_endpoint_error_x"].std(ddof=1)),
        "abs_endpoint_error_x_m_q25": float(valid["abs_endpoint_error_x"].quantile(0.25)),
        "abs_endpoint_error_x_m_median": float(valid["abs_endpoint_error_x"].median()),
        "abs_endpoint_error_x_m_q75": float(valid["abs_endpoint_error_x"].quantile(0.75)),
        "abs_endpoint_error_x_m_q90": float(valid["abs_endpoint_error_x"].quantile(0.90)),
        "abs_lateral_drift_rate_mean": float(valid["abs_lateral_drift_rate"].mean()),
        "abs_lateral_drift_rate_std": float(valid["abs_lateral_drift_rate"].std(ddof=1)),
        "abs_lateral_drift_rate_q25": float(valid["abs_lateral_drift_rate"].quantile(0.25)),
        "abs_lateral_drift_rate_median": float(valid["abs_lateral_drift_rate"].median()),
        "abs_lateral_drift_rate_q75": float(valid["abs_lateral_drift_rate"].quantile(0.75)),
        "abs_lateral_drift_rate_q90": float(valid["abs_lateral_drift_rate"].quantile(0.90)),
        "observed_path_length_scale_median": float(valid["path_length_scale"].median()),
    }
    return model, rows


def _apply_error_model(
    manifest: pd.DataFrame,
    model: dict[str, float | str | int],
    trials_dir: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    forward_scale = float(model["forward_scale_for_correction"])
    lateral_drift_rate = float(model["lateral_drift_rate_for_correction"])
    if forward_scale <= 0:
        raise ValueError("forward_scale must be positive")

    manifest_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for _, row in manifest.iterrows():
        original_path = Path(row["trajectory_csv"])
        trajectory = pd.read_csv(original_path)
        corrected = _correct_trajectory(trajectory, forward_scale, lateral_drift_rate)
        output_path = trials_dir / _relative_trial_path(row) / "trajectory.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        corrected.to_csv(output_path, index=False)

        before_endpoint = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        after_endpoint = corrected[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        endpoint_shift = after_endpoint - before_endpoint

        manifest_row = row.to_dict()
        manifest_row["trajectory_csv"] = str(output_path)
        manifest_row["endpoint_x"] = float(after_endpoint[0])
        manifest_row["endpoint_y"] = float(after_endpoint[1])
        manifest_row["processing_mode"] = (
            f"{row.get('processing_mode', '')}+straight_walk_error_correction"
        )
        manifest_rows.append(manifest_row)

        correction_rows.append(
            {
                "dataset": row.get("dataset"),
                "trajectory_name": row.get("trajectory_name"),
                "trial_id": row.get("trial_id"),
                "original_trajectory_csv": str(original_path),
                "corrected_trajectory_csv": str(output_path),
                "original_endpoint_x": float(before_endpoint[0]),
                "original_endpoint_y": float(before_endpoint[1]),
                "corrected_endpoint_x": float(after_endpoint[0]),
                "corrected_endpoint_y": float(after_endpoint[1]),
                "endpoint_shift_x": float(endpoint_shift[0]),
                "endpoint_shift_y": float(endpoint_shift[1]),
                "endpoint_shift_m": float(np.linalg.norm(endpoint_shift)),
                "forward_scale_for_correction": forward_scale,
                "lateral_drift_rate_for_correction": lateral_drift_rate,
                "abs_lateral_endpoint_uncertainty_median_m": float(
                    model["abs_endpoint_error_x_m_median"]
                ),
                "abs_lateral_endpoint_uncertainty_q75_m": float(
                    model["abs_endpoint_error_x_m_q75"]
                ),
            }
        )
        pairs.append(
            {
                "dataset": row.get("dataset"),
                "trajectory_name": row.get("trajectory_name"),
                "trial_id": row.get("trial_id"),
                "before": trajectory,
                "after": corrected,
            }
        )

    return pd.DataFrame(manifest_rows), correction_rows, pairs


def _correct_trajectory(
    trajectory: pd.DataFrame,
    forward_scale: float,
    lateral_drift_rate: float,
) -> pd.DataFrame:
    corrected = trajectory.copy()
    headings = trajectory["heading_rad"].to_numpy(dtype=float)
    observed_steps = trajectory["step_length_m"].to_numpy(dtype=float)
    true_steps = observed_steps / forward_scale

    forward_x = np.sin(headings)
    forward_y = np.cos(headings)
    right_x = np.cos(headings)
    right_y = -np.sin(headings)

    corrected_dx = true_steps * forward_x - lateral_drift_rate * true_steps * right_x
    corrected_dy = true_steps * forward_y - lateral_drift_rate * true_steps * right_y
    corrected["step_length_m_original"] = observed_steps
    corrected["step_length_m"] = true_steps
    corrected["x_original"] = trajectory["x"].to_numpy(dtype=float)
    corrected["y_original"] = trajectory["y"].to_numpy(dtype=float)
    corrected["x"] = np.cumsum(corrected_dx)
    corrected["y"] = np.cumsum(corrected_dy)
    corrected["straight_error_correction_forward_scale"] = forward_scale
    corrected["straight_error_correction_lateral_drift_rate"] = lateral_drift_rate
    return corrected


def _trajectory_path_length(trajectory: pd.DataFrame) -> float:
    if "step_length_m" in trajectory.columns:
        return float(trajectory["step_length_m"].sum())
    points = trajectory[["x", "y"]].to_numpy(dtype=float)
    points = np.vstack([np.zeros((1, 2)), points])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _trial_config_for_available_sensors(config: dict[str, Any], trial_dir: Path) -> dict[str, Any]:
    trial_config = deepcopy(config)
    has_gravity = (trial_dir / "Gravity.csv").exists()
    has_magnetometer = (trial_dir / "Magnetometer.csv").exists()
    if not has_gravity or not has_magnetometer:
        trial_config["preprocessing"]["step_signal"] = "acc_norm"
        trial_config["preprocessing"]["acc_smoothing_window"] = trial_config["comparison"]["simple"][
            "acc_smoothing_window"
        ]
        trial_config["step_detection"]["height"] = trial_config["comparison"]["simple"]["height"]
        trial_config["step_detection"]["distance_s"] = trial_config["comparison"]["simple"]["distance_s"]
        trial_config["step_detection"]["prominence"] = trial_config["comparison"]["simple"].get(
            "prominence"
        )
        trial_config["heading"]["use_smart_pdr"] = False
        trial_config["heading"]["bias_static_duration_s"] = trial_config["comparison"]["simple"][
            "bias_static_duration_s"
        ]
    return trial_config


def _processing_mode(config: dict[str, Any]) -> str:
    if bool(config["heading"].get("use_smart_pdr", False)):
        return "smartpdr_magnetic_no_table_correction"
    return "gyro_only_no_table_correction"


def _relative_trial_path(row: pd.Series) -> Path:
    trial_id = str(row["trial_id"]).strip("/")
    if trial_id:
        return Path(trial_id)
    return Path(str(row["trajectory_name"])) / Path(str(row.name))


def _trial_slug(path: Path, index: int) -> str:
    stem = path.stem.strip() or "trial"
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._")
    return f"{index:03d}_{slug or 'trial'}"


def _plot_straight_walks(
    rows: list[dict[str, Any]],
    true_distance_m: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for row in rows:
        trajectory = row["trajectory"]
        if trajectory.empty:
            continue
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            alpha=0.35,
            linewidth=1.0,
        )
        ax.scatter(
            [float(trajectory["x"].iloc[-1])],
            [float(trajectory["y"].iloc[-1])],
            marker="x",
            s=36,
            linewidths=1.0,
        )
    ax.plot([0.0, 0.0], [0.0, true_distance_m], color="black", linewidth=2.0, label="expected 10 m")
    ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="start")
    ax.scatter([0.0], [true_distance_m], marker="*", s=90, color="black", label="expected end")
    ax.set_title("10 m straight-walk correction trials")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_overall_pairs(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
    for ax, key, title in (
        (axes[0], "before", "Before correction"),
        (axes[1], "after", "After straight-walk error correction"),
    ):
        seen_labels: set[str] = set()
        for pair in pairs:
            label = str(pair["trajectory_name"])
            trajectory = pair[key]
            if trajectory.empty:
                continue
            legend_label = label if label not in seen_labels else None
            seen_labels.add(label)
            color = COLORS.get(label, "tab:gray")
            ax.plot(
                [0.0, *trajectory["x"].to_list()],
                [0.0, *trajectory["y"].to_list()],
                color=color,
                alpha=0.18,
                linewidth=1.0,
                label=legend_label,
            )
            ax.scatter(
                [float(trajectory["x"].iloc[-1])],
                [float(trajectory["y"].iloc[-1])],
                color=color,
                marker="x",
                alpha=0.75,
                s=36,
                linewidths=1.0,
            )
        _draw_table_a_overlay(ax, config)
        ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="start")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_shift(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, group in summary.groupby("trajectory_name"):
        ax.scatter(
            group["original_endpoint_x"],
            group["original_endpoint_y"],
            color=COLORS.get(str(label), "tab:gray"),
            alpha=0.25,
            s=22,
            label=f"{label} before",
        )
        ax.scatter(
            group["corrected_endpoint_x"],
            group["corrected_endpoint_y"],
            color=COLORS.get(str(label), "tab:gray"),
            alpha=0.85,
            marker="x",
            s=32,
            label=f"{label} after",
        )
    ax.set_title("Endpoint shift by straight-walk error correction")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="x-small", ncol=1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_table_a_overlay(ax: Any, config: dict[str, Any]) -> None:
    overlay = config.get("table_region", {}).get("table_overlays", {}).get("A")
    if not overlay:
        return
    left = float(overlay["left"])
    bottom = float(overlay["bottom"])
    width = float(overlay["width"])
    height = float(overlay["height"])
    rectangle = plt.Rectangle(
        (left, bottom),
        width,
        height,
        fill=False,
        edgecolor="tab:blue",
        linewidth=1.5,
        linestyle="--",
        alpha=0.8,
        label="table A reference",
    )
    ax.add_patch(rectangle)


if __name__ == "__main__":
    main()
