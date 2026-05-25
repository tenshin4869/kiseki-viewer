from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import (
    dataset_name,
    holding_position_from_trial,
    output_trial_id,
    read_accelerometer_csv,
    read_gyroscope_csv,
    resolve_trial_dir,
)
from .plotting import plot_acc_norm, plot_heading, plot_trajectory
from .processing import add_acc_norm, build_trajectory, detect_steps, estimate_heading


def run_trial(trial_id: str, config: dict[str, Any]) -> dict[str, Path]:
    raw_dir = resolve_trial_dir(config["paths"]["raw_data_dir"], trial_id)
    output_id = output_trial_id(config["paths"]["raw_data_dir"], raw_dir)
    output_root = Path(config["paths"]["output_dir"])
    acc_df = read_accelerometer_csv(raw_dir / "Accelerometer.csv")
    gyro_df = read_gyroscope_csv(raw_dir / "Gyroscope.csv")

    acc_df = add_acc_norm(acc_df, int(config["preprocessing"]["acc_smoothing_window"]))
    step_height = _step_detection_height_for_trial(config, raw_dir)
    steps_df = detect_steps(
        acc_df,
        height=step_height,
        distance_s=float(config["step_detection"]["distance_s"]),
        prominence=config["step_detection"].get("prominence"),
    )
    heading_df = estimate_heading(
        gyro_df,
        gyro_axis=str(config["heading"]["gyro_axis"]),
        gyro_sign=float(config["heading"]["gyro_sign"]),
        initial_heading_rad=float(config["heading"]["initial_heading_rad"]),
        use_bias_correction=bool(config["heading"].get("use_bias_correction", True)),
        bias_static_duration_s=float(config["heading"]["bias_static_duration_s"]),
    )
    step_length_m = _step_length_for_trial(config, raw_dir)
    trajectory_df = build_trajectory(steps_df, heading_df, step_length_m)

    processed_dir = output_root / output_id / "processed"
    figures_dir = output_root / output_id / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "steps": processed_dir / "steps.csv",
        "heading": processed_dir / "heading.csv",
        "trajectory": processed_dir / "trajectory.csv",
        "acc_norm_figure": figures_dir / "acc_norm.png",
        "heading_figure": figures_dir / "heading.png",
        "trajectory_figure": figures_dir / "trajectory.png",
    }
    steps_df.to_csv(paths["steps"], index=False)
    heading_df[["t", "heading_rad"]].to_csv(paths["heading"], index=False)
    trajectory_df.to_csv(paths["trajectory"], index=False)

    dpi = int(config["visualization"]["figure_dpi"])
    show_grid = bool(config["visualization"]["show_grid"])
    equal_axis = bool(config["visualization"]["equal_axis"])
    plot_acc_norm(acc_df, steps_df, paths["acc_norm_figure"], dpi, show_grid)
    plot_heading(heading_df, str(config["heading"]["gyro_axis"]), paths["heading_figure"], dpi, show_grid)
    plot_trajectory(trajectory_df, paths["trajectory_figure"], output_id, dpi, equal_axis, show_grid)
    return paths


def _step_length_for_trial(config: dict[str, Any], raw_dir: Path) -> float:
    pdr_config = config["pdr"]
    step_length_by_dataset = pdr_config.get("step_length_by_dataset", {})
    name = dataset_name(config["paths"]["raw_data_dir"], raw_dir)
    step_length_m = step_length_by_dataset.get(name, pdr_config["step_length_m"])
    return float(step_length_m)


def _step_detection_height_for_trial(config: dict[str, Any], raw_dir: Path) -> float | None:
    step_config = config["step_detection"]
    name = dataset_name(config["paths"]["raw_data_dir"], raw_dir)
    holding_position = holding_position_from_trial(raw_dir)
    height = step_config.get("height")
    by_dataset = step_config.get("height_by_dataset", {})
    dataset_height = by_dataset.get(name)
    if isinstance(dataset_height, dict):
        height = dataset_height.get(holding_position, height)
    elif dataset_height is not None:
        height = dataset_height
    return None if height is None else float(height)
