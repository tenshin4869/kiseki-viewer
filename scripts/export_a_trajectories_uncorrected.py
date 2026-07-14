from __future__ import annotations

import argparse
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

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.trial import run_trial


A_LABELS = ("A-1", "A-2", "A-3", "A-4")
A_FOLDER_ALIASES = {
    "A-1": "A-1",
    "A-2": "A-2",
    "A-3": "A-3",
    "A-4": "A-4",
    "A1": "A-1",
    "A2": "A-2",
    "A3": "A-3",
    "A4": "A-4",
}
COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export uncorrected A-table trajectories from 001-004 zip trials."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--raw-data-dir", default="data")
    parser.add_argument("--datasets", nargs="+", default=["001", "002", "003", "004"])
    parser.add_argument("--output-dir", default="output/a_trajectories_uncorrected")
    args = parser.parse_args()

    source_root = Path(args.raw_data_dir)
    output_root = Path(args.output_dir)
    extracted_root = output_root / "extracted_raw"
    trials_output_root = output_root / "trials"
    figures_dir = output_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    extracted_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    run_config = deepcopy(config)
    run_config["paths"]["raw_data_dir"] = str(extracted_root)
    run_config["paths"]["output_dir"] = str(trials_output_root)

    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []

    zip_paths = _find_a_zip_paths(source_root, args.datasets)
    for index, zip_path in enumerate(zip_paths, start=1):
        dataset = zip_path.relative_to(source_root).parts[0]
        trajectory_name = A_FOLDER_ALIASES[zip_path.parent.name]
        trial_slug = _trial_slug(zip_path, index)
        extracted_dir = extracted_root / dataset / trajectory_name / trial_slug
        extracted_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted_dir)

        trial_id = str(extracted_dir.relative_to(extracted_root))
        required = ["Accelerometer.csv", "Gyroscope.csv"]
        missing = [name for name in required if not (extracted_dir / name).exists()]
        if missing:
            skipped_rows.append(
                {
                    "source_path": str(zip_path),
                    "trajectory_name": trajectory_name,
                    "reason": f"missing required files: {', '.join(missing)}",
                }
            )
            continue

        trial_config = _trial_config_for_available_sensors(run_config, extracted_dir)
        try:
            outputs = run_trial(trial_id, trial_config)
            trajectory = pd.read_csv(outputs["trajectory"])
        except Exception as exc:  # noqa: BLE001 - keep processing the remaining experiments.
            skipped_rows.append(
                {
                    "source_path": str(zip_path),
                    "trajectory_name": trajectory_name,
                    "reason": f"processing failed: {exc}",
                }
            )
            continue

        trajectory_csv = outputs["trajectory"]
        manifest_rows.append(
            {
                "dataset": dataset,
                "trajectory_name": trajectory_name,
                "source_path": str(zip_path),
                "trial_id": trial_id,
                "step_count": int(len(trajectory)),
                "endpoint_x": float(trajectory["x"].iloc[-1]) if not trajectory.empty else None,
                "endpoint_y": float(trajectory["y"].iloc[-1]) if not trajectory.empty else None,
                "processing_mode": _processing_mode(trial_config),
                "trajectory_csv": str(trajectory_csv),
            }
        )
        trajectories.append(
            {
                "dataset": dataset,
                "trajectory_name": trajectory_name,
                "source_path": str(zip_path),
                "trajectory": trajectory,
            }
        )

    _record_accelerometer_only_files(source_root, args.datasets, skipped_rows)

    manifest = pd.DataFrame(manifest_rows)
    skipped = pd.DataFrame(skipped_rows)
    manifest_path = output_root / "manifest.csv"
    skipped_path = output_root / "skipped.csv"
    manifest.to_csv(manifest_path, index=False)
    skipped.to_csv(skipped_path, index=False)

    _plot_all_a_trajectories(
        trajectories,
        figures_dir / "a_trajectories_uncorrected_all.png",
        "A trajectories 001-004 (uncorrected)",
        config,
    )
    for dataset in args.datasets:
        dataset_trajectories = [row for row in trajectories if row["dataset"] == dataset]
        if dataset_trajectories:
            _plot_all_a_trajectories(
                dataset_trajectories,
                figures_dir / f"a_trajectories_uncorrected_{dataset}.png",
                f"A trajectories {dataset} (uncorrected)",
                config,
            )

    print(f"Processed zip trials: {len(manifest_rows)}")
    print(f"Skipped files: {len(skipped_rows)}")
    print(f"Manifest: {manifest_path}")
    print(f"Skipped: {skipped_path}")
    print(f"All A trajectory figure: {figures_dir / 'a_trajectories_uncorrected_all.png'}")
    for dataset in args.datasets:
        path = figures_dir / f"a_trajectories_uncorrected_{dataset}.png"
        if path.exists():
            print(f"{dataset} figure: {path}")


def _find_a_zip_paths(source_root: Path, datasets: list[str]) -> list[Path]:
    paths: list[Path] = []
    for dataset in datasets:
        for folder_name in A_FOLDER_ALIASES:
            paths.extend(sorted((source_root / dataset / folder_name).glob("*.zip")))
    return sorted(paths)


def _record_accelerometer_only_files(
    source_root: Path, datasets: list[str], skipped_rows: list[dict[str, Any]]
) -> None:
    for dataset in datasets:
        for folder_name, label in A_FOLDER_ALIASES.items():
            folder = source_root / dataset / folder_name
            for path in sorted(folder.glob("*.xls")) + sorted(folder.glob("*.csv")):
                if path.name in {"Accelerometer.csv", "Gyroscope.csv", "Gravity.csv", "Magnetometer.csv"}:
                    continue
                if path.suffix.lower() == ".csv" and path.with_suffix(".xls").exists():
                    reason = "converted from .xls, but only contains one sensor table"
                elif path.suffix.lower() == ".xls":
                    reason = "Excel file is not a complete PDR trial archive"
                else:
                    continue
                skipped_rows.append(
                    {
                        "source_path": str(path),
                        "trajectory_name": label,
                        "reason": reason,
                    }
                )


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


def _trial_slug(path: Path, index: int) -> str:
    stem = path.stem.strip() or "trial"
    slug = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("._")
    return f"{index:03d}_{slug or 'trial'}"


def _plot_all_a_trajectories(
    rows: list[dict[str, Any]],
    output_path: Path,
    title: str,
    config: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    seen_labels: set[str] = set()
    for row in rows:
        label = str(row["trajectory_name"])
        trajectory = row["trajectory"]
        if trajectory.empty:
            continue
        legend_label = label if label not in seen_labels else None
        seen_labels.add(label)
        ax.plot(
            [0.0, *trajectory["x"].to_list()],
            [0.0, *trajectory["y"].to_list()],
            color=COLORS.get(label, "tab:gray"),
            alpha=0.20,
            linewidth=1.0,
            label=legend_label,
            zorder=2,
        )
        ax.scatter(
            trajectory["x"],
            trajectory["y"],
            color=COLORS.get(label, "tab:gray"),
            alpha=0.26,
            s=8,
            linewidths=0,
            zorder=3,
        )
        ax.scatter(
            [float(trajectory["x"].iloc[-1])],
            [float(trajectory["y"].iloc[-1])],
            color=COLORS.get(label, "tab:gray"),
            marker="x",
            alpha=0.90,
            s=44,
            linewidths=1.2,
            zorder=8,
        )
    _draw_table_a_overlay(ax, config)
    ax.scatter([0.0], [0.0], marker="s", s=70, color="tab:green", label="start", zorder=10)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if bool(config["visualization"].get("equal_axis", True)):
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(bool(config["visualization"].get("show_grid", True)))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["visualization"].get("figure_dpi", 200)))
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
