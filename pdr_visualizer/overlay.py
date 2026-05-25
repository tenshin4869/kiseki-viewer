from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import dataset_name, find_trial_dirs, holding_position_from_trial, output_trial_id
from .plotting import plot_overlay
from .trial import run_trial


def plot_all_trials(config: dict[str, Any]) -> dict[str, Path]:
    raw_data_dir = Path(config["paths"]["raw_data_dir"])
    output_root = Path(config["paths"]["output_dir"])
    all_trajectories: list[tuple[str, pd.DataFrame]] = []
    grouped: dict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = {}
    condition_grouped: dict[str, list[tuple[str, pd.DataFrame]]] = {"hand": [], "pocket": []}

    for trial_path in find_trial_dirs(raw_data_dir):
        trial_id = output_trial_id(raw_data_dir, trial_path)
        trajectory_path = output_root / trial_id / "processed" / "trajectory.csv"
        if not trajectory_path.exists():
            run_trial(trial_id, config)
        trajectory = pd.read_csv(trajectory_path)
        label = f"{trial_id} ({holding_position_from_trial(trial_path)})"
        all_trajectories.append((label, trajectory))
        condition = holding_position_from_trial(trial_path)
        dataset = dataset_name(raw_data_dir, trial_path)
        condition_grouped[condition].append((trial_id, trajectory))
        grouped.setdefault((dataset, condition), []).append((trial_path.name, trajectory))

    if not all_trajectories:
        raise ValueError(f"No trials found under {raw_data_dir}")

    outputs: dict[str, Path] = {}
    outputs["all"] = _plot_group(config, all_trajectories, output_root / "overlays" / "all" / "trajectory_overlay.png", "all trials")
    for condition, trajectories in condition_grouped.items():
        if trajectories:
            outputs[condition] = _plot_group(
                config,
                trajectories,
                output_root / "overlays" / condition / "trajectory_overlay.png",
                f"{condition} trials",
            )
    for (dataset, condition), trajectories in sorted(grouped.items()):
        outputs[f"{dataset}_{condition}"] = _plot_group(
            config,
            trajectories,
            output_root / "overlays" / dataset / condition / "trajectory_overlay.png",
            f"{dataset} {condition} trials",
        )
    return outputs


def _plot_group(
    config: dict[str, Any],
    trajectories: list[tuple[str, pd.DataFrame]],
    output_path: Path,
    title: str,
) -> Path:
    plot_overlay(
        trajectories,
        output_path,
        title=title,
        dpi=int(config["visualization"]["figure_dpi"]),
        equal_axis=bool(config["visualization"]["equal_axis"]),
        show_grid=bool(config["visualization"]["show_grid"]),
    )
    return output_path
