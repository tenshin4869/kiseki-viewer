from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.io import find_trial_dirs, holding_position_from_trial
from pdr_visualizer.plotting import plot_overlay
from pdr_visualizer.trial import run_trial


METHODS = {
    "simple_legacy": "Simple (legacy)",
    "smartpdr_magnetic": "SmartPDR + magnetic",
    "smartpdr_no_magnetic": "SmartPDR without magnetic",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export hand and pocket overlay figures for the three comparison methods."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output_test/trajectory_overlays")
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Only export selected datasets, for example: --datasets Data5 Data6",
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(args.output_dir)
    raw_root = Path(base_config["paths"]["raw_data_dir"])
    trial_ids = [str(trial_path.relative_to(raw_root)) for trial_path in find_trial_dirs(raw_root)]
    if args.datasets:
        selected_datasets = set(args.datasets)
        trial_ids = [
            trial_id for trial_id in trial_ids if trial_id.split("/", 1)[0] in selected_datasets
        ]
    if not trial_ids:
        raise ValueError("No trials found for the selected datasets.")
    trajectories: dict[str, dict[str, pd.DataFrame]] = {method: {} for method in METHODS}

    magnetic_config = _method_config(base_config, output_root / "smartpdr_magnetic", 0.05)
    no_magnetic_config = _method_config(base_config, output_root / "smartpdr_no_magnetic", 0.0)
    for trial_id in trial_ids:
        magnetic_paths = run_trial(trial_id, magnetic_config)
        no_magnetic_paths = run_trial(trial_id, no_magnetic_config)
        trajectories["smartpdr_magnetic"][trial_id] = pd.read_csv(magnetic_paths["trajectory"])
        trajectories["smartpdr_no_magnetic"][trial_id] = pd.read_csv(no_magnetic_paths["trajectory"])
        simple = pd.read_csv(magnetic_paths["simple_trajectory"])
        trajectories["simple_legacy"][trial_id] = simple
        simple_path = output_root / "simple_legacy" / trial_id / "processed" / "trajectory.csv"
        simple_path.parent.mkdir(parents=True, exist_ok=True)
        simple.to_csv(simple_path, index=False)

    _write_overlays(output_root, base_config, trajectories)
    _write_readme(output_root, trajectories)
    print(f"output: {output_root}")
    for method in METHODS:
        for condition in ("hand", "pocket"):
            path = output_root / method / condition / "trajectory_overlay.png"
            if path.exists():
                print(f"{method} {condition}: {path}")
        for dataset in sorted({trial_id.split("/", 1)[0] for trial_id in trial_ids}):
            for condition in ("hand", "pocket"):
                path = output_root / method / dataset / condition / "trajectory_overlay.png"
                if path.exists():
                    print(f"{method} {dataset} {condition}: {path}")


def _method_config(base_config: dict, output_dir: Path, magnetic_gain: float) -> dict:
    config = deepcopy(base_config)
    config["paths"]["output_dir"] = str(output_dir)
    config["heading"]["initial_alignment"]["enabled"] = False
    config["heading"]["mag_correction_gain"] = magnetic_gain
    return config


def _write_overlays(
    output_root: Path,
    config: dict,
    trajectories: dict[str, dict[str, pd.DataFrame]],
) -> None:
    dpi = int(config["visualization"]["figure_dpi"])
    equal_axis = bool(config["visualization"]["equal_axis"])
    show_grid = bool(config["visualization"]["show_grid"])
    for method, label in METHODS.items():
        datasets = sorted({trial_id.split("/", 1)[0] for trial_id in trajectories[method]})
        scope = " + ".join(datasets) if len(datasets) <= 2 else ""
        for condition in ("hand", "pocket"):
            selected = [
                (trial_id, trajectory)
                for trial_id, trajectory in trajectories[method].items()
                if holding_position_from_trial(trial_id) == condition
            ]
            if not selected:
                continue
            plot_overlay(
                selected,
                output_root / method / condition / "trajectory_overlay.png",
                title=f"{label}: {scope + ' ' if scope else ''}{condition} trials",
                dpi=dpi,
                equal_axis=equal_axis,
                show_grid=show_grid,
            )
        for dataset in datasets:
            for condition in ("hand", "pocket"):
                selected = [
                    (trial_id, trajectory)
                    for trial_id, trajectory in trajectories[method].items()
                    if trial_id.startswith(f"{dataset}/")
                    and holding_position_from_trial(trial_id) == condition
                ]
                if not selected:
                    continue
                plot_overlay(
                    selected,
                    output_root / method / dataset / condition / "trajectory_overlay.png",
                    title=f"{label}: {dataset} {condition} trials",
                    dpi=dpi,
                    equal_axis=equal_axis,
                    show_grid=show_grid,
                )


def _write_readme(output_root: Path, trajectories: dict[str, dict[str, pd.DataFrame]]) -> None:
    hand_count = sum(
        holding_position_from_trial(trial_id) == "hand"
        for trial_id in trajectories["simple_legacy"]
    )
    pocket_count = sum(
        holding_position_from_trial(trial_id) == "pocket"
        for trial_id in trajectories["simple_legacy"]
    )
    datasets = sorted({trial_id.split("/", 1)[0] for trial_id in trajectories["simple_legacy"]})
    dataset_text = ", ".join(f"`{dataset}`" for dataset in datasets)
    text = f"""# 3方式の全体軌跡比較

## 出力内容

| フォルダ | 方位・処理条件 |
| --- | --- |
| `simple_legacy` | SmartPDR再現実装を用いないシンプル版 |
| `smartpdr_magnetic` | SmartPDR風再現実装、磁気補正あり (`mag_correction_gain=0.05`) |
| `smartpdr_no_magnetic` | SmartPDR風再現実装、磁気補正なし (`mag_correction_gain=0.0`) |

各方式で、対象データに存在する条件の `hand/trajectory_overlay.png` または `pocket/trajectory_overlay.png` を生成した。
対象試行数は `hand={hand_count}`、`pocket={pocket_count}` である。
対象データセットは {dataset_text} である。

## 比較上の注意

- `smartpdr_magnetic` と `smartpdr_no_magnetic` の差は、磁気による方位補正の有無だけである。
- `simple_legacy` と SmartPDR風再現実装の差には、方位だけでなく歩行信号処理やステップ検出設定の差も含まれる。
- すべての SmartPDR風出力で、最初の歩数を正面へ揃える初期アライメントは無効である。

## データセット別の図

各方式の直下に `<dataset>/<condition>/trajectory_overlay.png` も生成する。
"""
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
