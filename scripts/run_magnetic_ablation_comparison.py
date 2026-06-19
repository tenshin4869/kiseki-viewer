from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.io import find_trial_dirs
from pdr_visualizer.overlay import plot_all_trials
from pdr_visualizer.trial import run_trial


VARIANTS = {
    "with_magnetic": {
        "label": "with magnetic correction",
        "gain": None,
        "color": "tab:blue",
    },
    "without_magnetic": {
        "label": "without magnetic correction",
        "gain": 0.0,
        "color": "tab:orange",
    },
}
STRAIGHT_TARGETS = {
    "Data2/001_walk": np.array([0.0, 6.0]),
    "Data3/Walk_002": np.array([0.0, 6.0]),
}
LOOP_TRIALS = {
    "Data2/LTurn_001",
    "Data2/RTurn_001",
    "Data3/LTurn_002",
    "Data3/RTurn_002",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablate only magnetic correction from the SmartPDR-like heading pipeline."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output_test/magnetic_ablation")
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(args.output_dir)
    raw_root = Path(base_config["paths"]["raw_data_dir"])
    trial_ids = [
        str(trial_path.relative_to(raw_root)) for trial_path in find_trial_dirs(raw_root)
    ]
    rows: list[dict[str, object]] = []
    trajectories: dict[str, dict[str, pd.DataFrame]] = {}

    for variant, details in VARIANTS.items():
        config = deepcopy(base_config)
        config["paths"]["output_dir"] = str(output_root / "variants" / variant)
        config["heading"]["initial_alignment"]["enabled"] = False
        if details["gain"] is not None:
            config["heading"]["mag_correction_gain"] = float(details["gain"])
        effective_gain = float(config["heading"]["mag_correction_gain"])
        trajectories[variant] = {}
        for trial_id in trial_ids:
            paths = run_trial(trial_id, config)
            trajectory = pd.read_csv(paths["trajectory"])
            heading = pd.read_csv(paths["heading"])
            trajectories[variant][trial_id] = trajectory
            rows.append(_summarize(trial_id, variant, details, effective_gain, trajectory, heading))
        plot_all_trials(config)

    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "magnetic_ablation_summary.csv", index=False)
    _plot_trajectory_comparisons(output_root, trajectories)
    _plot_straight_x_displacement(output_root, summary)
    _plot_loop_closure(output_root, summary)
    _write_report(output_root, summary)
    print(f"summary: {output_root / 'magnetic_ablation_summary.csv'}")
    print(f"report: {output_root / 'magnetic_ablation_report.md'}")
    print(f"comparisons: {output_root / 'comparisons'}")


def _summarize(
    trial_id: str,
    variant: str,
    details: dict[str, object],
    magnetic_gain: float,
    trajectory: pd.DataFrame,
    heading: pd.DataFrame,
) -> dict[str, object]:
    end = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
    target = STRAIGHT_TARGETS.get(trial_id)
    heading_values = np.unwrap(heading["heading_rad"].to_numpy(dtype=float))
    gyro_values = np.unwrap(heading["heading_gyro_rad"].to_numpy(dtype=float))
    magnetic_effect = np.degrees(
        np.angle(
            np.exp(
                1j
                * (
                    heading["heading_rad"].to_numpy(dtype=float)
                    - heading["heading_gyro_rad"].to_numpy(dtype=float)
                )
            )
        )
    )
    return {
        "trial_id": trial_id,
        "variant": variant,
        "variant_label": details["label"],
        "mag_correction_gain": magnetic_gain,
        "steps": len(trajectory),
        "distance_m": float(trajectory["step_length_m"].sum()),
        "end_x_m": float(end[0]),
        "end_y_m": float(end[1]),
        "straight_endpoint_error_m": (
            float(np.linalg.norm(end - target)) if target is not None else np.nan
        ),
        "loop_closure_error_m": float(np.linalg.norm(end)) if trial_id in LOOP_TRIALS else np.nan,
        "heading_change_deg": float(np.degrees(heading_values[-1] - heading_values[0])),
        "gyro_heading_change_deg": float(np.degrees(gyro_values[-1] - gyro_values[0])),
        "max_abs_magnetic_heading_effect_deg": float(np.max(np.abs(magnetic_effect))),
        "mag_acceptance_pct": float(heading["mag_validation_accepted"].mean() * 100.0),
    }


def _plot_trajectory_comparisons(
    output_root: Path, trajectories: dict[str, dict[str, pd.DataFrame]]
) -> None:
    trial_ids = next(iter(trajectories.values()))
    for trial_id in trial_ids:
        fig, ax = plt.subplots(figsize=(8, 8))
        for variant, details in VARIANTS.items():
            trajectory = trajectories[variant][trial_id]
            x = [0.0, *trajectory["x"].to_list()]
            y = [0.0, *trajectory["y"].to_list()]
            ax.plot(x, y, marker="o", markersize=3, color=details["color"], label=details["label"])
            ax.scatter([x[-1]], [y[-1]], marker="x", s=50, color=details["color"])
        ax.scatter([0.0], [0.0], marker="s", s=50, color="black", label="start", zorder=4)
        ax.set_title(f"{trial_id}: effect of magnetic correction only")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend()
        _save(fig, output_root / "comparisons" / trial_id / "magnetic_ablation.png")


def _plot_straight_x_displacement(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(STRAIGHT_TARGETS)]
    pivot = selection.pivot(index="trial_id", columns="variant", values="end_x_m")
    pivot = pivot[["without_magnetic", "with_magnetic"]]
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["tab:orange", "tab:blue"])
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Endpoint x [m]")
    ax.set_xlabel("")
    ax.set_title("Straight trials: x displacement caused by magnetic correction")
    ax.legend(
        [ax.containers[0], ax.containers[1]],
        ["without magnetic correction", "with magnetic correction"],
    )
    ax.grid(True, axis="y")
    _save(ax.figure, output_root / "metrics" / "straight_endpoint_x.png")


def _plot_loop_closure(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(LOOP_TRIALS)]
    pivot = selection.pivot(index="trial_id", columns="variant", values="loop_closure_error_m")
    pivot = pivot[["without_magnetic", "with_magnetic"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["tab:orange", "tab:blue"])
    ax.set_ylabel("Closure error [m]")
    ax.set_xlabel("")
    ax.set_title("Loop trials: effect of magnetic correction only")
    ax.legend(["without magnetic correction", "with magnetic correction"])
    ax.grid(True, axis="y")
    _save(ax.figure, output_root / "metrics" / "loop_closure_error.png")


def _write_report(output_root: Path, summary: pd.DataFrame) -> None:
    straight = summary[summary["trial_id"].isin(STRAIGHT_TARGETS)]
    loops = summary[summary["trial_id"].isin(LOOP_TRIALS)]
    lines = [
        "# SmartPDR風処理における磁気補正のみの有無比較",
        "",
        "## 検証条件",
        "",
        "今回の比較では、歩数検出、歩幅、加速度処理、重力方向へ射影したジャイロ、バイアス補正、初期アライメント無効を共通にした。変更した値は `mag_correction_gain` だけである。",
        "",
        "| 条件 | `mag_correction_gain` | 意味 |",
        "| --- | ---: | --- |",
        "| `with magnetic correction` | `0.05` | 現在の磁気・ジャイロ融合 |",
        "| `without magnetic correction` | `0.0` | 磁気が方位へ作用しない、重力軸ジャイロのみの SmartPDR 処理 |",
        "",
        "磁気なし条件でも診断用に磁気CSVは読み込むが、ゲインが0であるため軌跡計算への磁気補正量は常に0となる。",
        "",
        "## 直進試行",
        "",
        "| 試行 | 磁気なし終点 x [m] | 磁気あり終点 x [m] | 磁気による x 差 [m] | 磁気なし終点誤差 [m] | 磁気あり終点誤差 [m] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trial_id in STRAIGHT_TARGETS:
        values = straight[straight["trial_id"] == trial_id].set_index("variant")
        no_mag = values.loc["without_magnetic"]
        with_mag = values.loc["with_magnetic"]
        lines.append(
            f"| {trial_id} | {no_mag['end_x_m']:.3f} | {with_mag['end_x_m']:.3f} | "
            f"{with_mag['end_x_m'] - no_mag['end_x_m']:.3f} | "
            f"{no_mag['straight_endpoint_error_m']:.3f} | {with_mag['straight_endpoint_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 周回試行",
            "",
            "| 試行 | 磁気なし閉路誤差 [m] | 磁気あり閉路誤差 [m] | 磁気なし方位変化 [deg] | 磁気あり方位変化 [deg] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial_id in sorted(LOOP_TRIALS):
        values = loops[loops["trial_id"] == trial_id].set_index("variant")
        no_mag = values.loc["without_magnetic"]
        with_mag = values.loc["with_magnetic"]
        lines.append(
            f"| {trial_id} | {no_mag['loop_closure_error_m']:.3f} | {with_mag['loop_closure_error_m']:.3f} | "
            f"{no_mag['heading_change_deg']:.2f} | {with_mag['heading_change_deg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "- 直進試行で磁気ありの `x` が磁気なしより負方向へ移動する場合、斜めずれは磁気補正によって増えたと切り分けられる。",
            "- 周回試行で磁気ありの閉路誤差が小さくても、総方位変化が理論値から遠ざかる場合は、磁気が正しく旋回を推定したのではなく、偶然終点を近づけた可能性も残る。",
            "- この比較により磁気の寄与を確認した後でのみ、磁場強度や不整合に応じた磁気ゲートを次の単独要因として検証すべきである。",
            "",
            "## 生成物",
            "",
            "- 試行別比較図: `output_test/magnetic_ablation/comparisons/<dataset>/<trial>/magnetic_ablation.png`",
            "- 直進 x 終点比較: `output_test/magnetic_ablation/metrics/straight_endpoint_x.png`",
            "- 周回閉路誤差比較: `output_test/magnetic_ablation/metrics/loop_closure_error.png`",
            "- 数値一覧: `output_test/magnetic_ablation/magnetic_ablation_summary.csv`",
            "- 条件別出力: `output_test/magnetic_ablation/variants/<variant>/...`",
        ]
    )
    (output_root / "magnetic_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
