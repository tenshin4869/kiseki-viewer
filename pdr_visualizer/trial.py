from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import pandas as pd

from .io import (
    dataset_name,
    holding_position_from_trial,
    output_trial_id,
    read_accelerometer_csv,
    read_gravity_csv,
    read_gyroscope_csv,
    read_magnetometer_csv,
    resolve_trial_dir,
)
from .plotting import plot_acc_norm, plot_heading, plot_trajectory, plot_trajectory_comparison
from .processing import add_acc_norm, add_step_lengths, build_trajectory, detect_steps, estimate_heading


def run_trial(trial_id: str, config: dict[str, Any]) -> dict[str, Path]:
    raw_dir = resolve_trial_dir(config["paths"]["raw_data_dir"], trial_id)
    output_id = output_trial_id(config["paths"]["raw_data_dir"], raw_dir)
    output_root = Path(config["paths"]["output_dir"])
    acc_df = read_accelerometer_csv(raw_dir / "Accelerometer.csv")
    gyro_df = read_gyroscope_csv(raw_dir / "Gyroscope.csv")
    gravity_df = read_gravity_csv(raw_dir / "Gravity.csv") if _uses_gravity(config) else None
    mag_df = read_magnetometer_csv(raw_dir / "Magnetometer.csv") if config["heading"].get("use_smart_pdr", False) else None

    acc_df = add_acc_norm(
        acc_df,
        window_size=int(config["preprocessing"]["acc_smoothing_window"]),
        step_signal=str(config["preprocessing"].get("step_signal", "acc_norm")),
        hpf_alpha=float(config["preprocessing"].get("hpf_alpha", 0.9)),
        gravity_df=gravity_df,
    )
    step_height = _step_detection_height_for_trial(config, raw_dir)
    steps_df = detect_steps(
        acc_df,
        height=step_height,
        distance_s=float(config["step_detection"]["distance_s"]),
        prominence=config["step_detection"].get("prominence"),
        use_custom_algorithm=bool(config["step_detection"].get("use_custom_algorithm", False)),
        window_n=int(config["step_detection"].get("window_n", 6)),
        peak_threshold=float(config["step_detection"].get("peak_threshold", 0.5)),
        pp_threshold=float(config["step_detection"].get("pp_threshold", 1.0)),
    )
    heading_df = estimate_heading(
        gyro_df,
        gyro_axis=str(config["heading"]["gyro_axis"]),
        gyro_sign=float(config["heading"]["gyro_sign"]),
        gyro_scale=float(config["heading"].get("gyro_scale", 1.0)),
        initial_heading_rad=float(config["heading"]["initial_heading_rad"]),
        use_bias_correction=bool(config["heading"].get("use_bias_correction", True)),
        bias_static_duration_s=float(config["heading"]["bias_static_duration_s"]),
        use_smart_pdr=bool(config["heading"].get("use_smart_pdr", False)),
        gravity_df=gravity_df,
        mag_df=mag_df,
        declination_rad=math.radians(float(config["heading"].get("declination_deg", 0.0))),
        h_cor_t_rad=math.radians(float(config["heading"].get("h_cor_t_deg", 0.0))),
        h_mag_t_rad=math.radians(float(config["heading"].get("h_mag_t_deg", 0.0))),
        mag_correction_gain=float(config["heading"].get("mag_correction_gain", 0.05)),
    )
    step_length_m = _step_length_for_trial(config, raw_dir)
    steps_df = add_step_lengths(
        steps_df,
        acc_df,
        mode=str(config["pdr"].get("step_length_mode", "fixed")),
        fixed_step_length_m=step_length_m,
        dynamic_config=config["pdr"].get("dynamic_step_length", {}),
    )
    trajectory_df = build_trajectory(steps_df, heading_df)
    simple_trajectory_df = _build_simple_trajectory(acc_df, gyro_df, raw_dir, config)

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
        "trajectory_comparison_figure": figures_dir / "trajectory_comparison.png",
    }
    steps_df.to_csv(paths["steps"], index=False)
    heading_df.to_csv(paths["heading"], index=False)
    trajectory_df.to_csv(paths["trajectory"], index=False)

    dpi = int(config["visualization"]["figure_dpi"])
    show_grid = bool(config["visualization"]["show_grid"])
    equal_axis = bool(config["visualization"]["equal_axis"])
    plot_acc_norm(acc_df, steps_df, paths["acc_norm_figure"], dpi, show_grid)
    plot_heading(heading_df, str(config["heading"]["gyro_axis"]), paths["heading_figure"], dpi, show_grid)
    plot_trajectory(trajectory_df, paths["trajectory_figure"], output_id, dpi, equal_axis, show_grid)
    plot_trajectory_comparison(
        trajectory_df,
        simple_trajectory_df,
        paths["trajectory_comparison_figure"],
        output_id,
        dpi,
        equal_axis,
        show_grid,
    )
    return paths


def _uses_gravity(config: dict[str, Any]) -> bool:
    return (
        str(config["preprocessing"].get("step_signal", "acc_norm")) == "vertical_hpf"
        or bool(config["heading"].get("use_smart_pdr", False))
    )


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


def _build_simple_trajectory(
    acc_df: pd.DataFrame,
    gyro_df: pd.DataFrame,
    raw_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    simple = config["comparison"]["simple"]
    simple_acc_df = add_acc_norm(
        acc_df,
        window_size=int(simple["acc_smoothing_window"]),
        step_signal="acc_norm",
    )
    simple_steps_df = detect_steps(
        simple_acc_df,
        height=_simple_value_for_trial(simple, "height", raw_dir, config),
        distance_s=float(simple["distance_s"]),
        prominence=simple.get("prominence"),
    )
    simple_heading_df = estimate_heading(
        gyro_df,
        gyro_axis=str(simple["gyro_axis"]),
        gyro_sign=float(simple["gyro_sign"]),
        gyro_scale=float(simple.get("gyro_scale", 1.0)),
        initial_heading_rad=float(simple["initial_heading_rad"]),
        use_bias_correction=bool(simple.get("use_bias_correction", True)),
        bias_static_duration_s=float(simple["bias_static_duration_s"]),
    )
    simple_steps_df = add_step_lengths(
        simple_steps_df,
        simple_acc_df,
        mode="fixed",
        fixed_step_length_m=_simple_value_for_trial(
            simple, "step_length_m", raw_dir, config
        ),
        dynamic_config={},
    )
    return build_trajectory(simple_steps_df, simple_heading_df)


def _simple_value_for_trial(
    simple: dict[str, Any], key: str, raw_dir: Path, config: dict[str, Any]
) -> float:
    value = simple[key]
    by_dataset = simple.get(f"{key}_by_dataset", {})
    name = dataset_name(config["paths"]["raw_data_dir"], raw_dir)
    dataset_value = by_dataset.get(name)
    if key == "height" and isinstance(dataset_value, dict):
        value = dataset_value.get(holding_position_from_trial(raw_dir), value)
    elif dataset_value is not None:
        value = dataset_value
    return float(value)
