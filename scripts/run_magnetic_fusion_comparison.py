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
        description="Compare only the legacy trajectory with magnetic/gyro fusion without initial alignment."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output_test/magnetic_fusion")
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    output_root = Path(args.output_dir)
    config["paths"]["output_dir"] = str(output_root)
    config["heading"]["initial_alignment"]["enabled"] = False

    rows: list[dict[str, object]] = []
    raw_root = Path(config["paths"]["raw_data_dir"])
    for trial_path in find_trial_dirs(raw_root):
        trial_id = str(trial_path.relative_to(raw_root))
        paths = run_trial(trial_id, config)
        fusion = pd.read_csv(paths["trajectory"])
        simple = pd.read_csv(paths["simple_trajectory"])
        heading = pd.read_csv(paths["heading"])
        rows.extend(_summarize_trajectory(trial_id, "magnetic_gyro_fusion", fusion, heading))
        rows.extend(_summarize_trajectory(trial_id, "simple_legacy", simple, None))

    plot_all_trials(config)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_root / "comparison_summary.csv", index=False)
    _plot_straight_errors(output_root, summary)
    _plot_loop_closure(output_root, summary)
    _plot_magnetic_acceptance(output_root, summary)
    _write_report(output_root, summary)
    print(f"summary: {output_root / 'comparison_summary.csv'}")
    print(f"report: {output_root / 'magnetic_fusion_report.md'}")
    print(f"figures: {output_root}")


def _summarize_trajectory(
    trial_id: str,
    method: str,
    trajectory: pd.DataFrame,
    heading: pd.DataFrame | None,
) -> list[dict[str, object]]:
    end = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
    target = STRAIGHT_TARGETS.get(trial_id)
    row: dict[str, object] = {
        "trial_id": trial_id,
        "method": method,
        "steps": len(trajectory),
        "distance_m": float(trajectory["step_length_m"].sum()),
        "end_x_m": float(end[0]),
        "end_y_m": float(end[1]),
        "straight_endpoint_error_m": (
            float(np.linalg.norm(end - target)) if target is not None else np.nan
        ),
        "loop_closure_error_m": float(np.linalg.norm(end)) if trial_id in LOOP_TRIALS else np.nan,
        "mag_acceptance_pct": np.nan,
        "mag_norm_median_ut": np.nan,
        "mag_norm_max_deviation_ut": np.nan,
        "heading_change_deg": np.nan,
        "gyro_heading_change_deg": np.nan,
    }
    if heading is not None:
        row.update(
            {
                "mag_acceptance_pct": float(heading["mag_validation_accepted"].mean() * 100.0),
                "mag_norm_median_ut": float(heading["mag_norm_ut"].median()),
                "mag_norm_max_deviation_ut": float(heading["mag_norm_deviation_ut"].max()),
                "heading_change_deg": _heading_change_deg(heading["heading_rad"]),
                "gyro_heading_change_deg": _heading_change_deg(heading["heading_gyro_rad"]),
            }
        )
    return [row]


def _heading_change_deg(values: pd.Series) -> float:
    unwrapped = np.unwrap(values.to_numpy(dtype=float))
    return float(np.degrees(unwrapped[-1] - unwrapped[0]))


def _plot_straight_errors(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(STRAIGHT_TARGETS)].copy()
    pivot = selection.pivot(index="trial_id", columns="method", values="straight_endpoint_error_m")
    pivot = pivot[["simple_legacy", "magnetic_gyro_fusion"]]
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["tab:orange", "tab:blue"])
    ax.set_ylabel("Endpoint error to known straight target [m]")
    ax.set_xlabel("")
    ax.set_title("Straight trials: simple vs magnetic/gyro fusion")
    ax.legend(["simple (legacy)", "magnetic/gyro fusion"])
    ax.grid(True, axis="y")
    _save(ax.figure, output_root / "metrics" / "straight_endpoint_error.png")


def _plot_loop_closure(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(LOOP_TRIALS)].copy()
    pivot = selection.pivot(index="trial_id", columns="method", values="loop_closure_error_m")
    pivot = pivot[["simple_legacy", "magnetic_gyro_fusion"]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["tab:orange", "tab:blue"])
    ax.set_ylabel("Closure error [m]")
    ax.set_xlabel("")
    ax.set_title("Loop trials: simple vs magnetic/gyro fusion")
    ax.legend(["simple (legacy)", "magnetic/gyro fusion"])
    ax.grid(True, axis="y")
    _save(ax.figure, output_root / "metrics" / "loop_closure_error.png")


def _plot_magnetic_acceptance(output_root: Path, summary: pd.DataFrame) -> None:
    fusion = summary[summary["method"] == "magnetic_gyro_fusion"].set_index("trial_id")
    selected_ids = [*STRAIGHT_TARGETS, *sorted(LOOP_TRIALS)]
    fusion = fusion.loc[selected_ids]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].bar(fusion.index, fusion["mag_acceptance_pct"], color="tab:blue")
    axes[0].set_ylabel("Accepted [%]")
    axes[0].set_title("Magnetic fusion diagnostics")
    axes[0].grid(True, axis="y")
    axes[1].bar(fusion.index, fusion["mag_norm_max_deviation_ut"], color="tab:red")
    axes[1].set_ylabel("Max deviation [uT]")
    axes[1].grid(True, axis="y")
    axes[1].tick_params(axis="x", rotation=20)
    _save(fig, output_root / "metrics" / "magnetic_diagnostics.png")


def _write_report(output_root: Path, summary: pd.DataFrame) -> None:
    straight = summary[summary["trial_id"].isin(STRAIGHT_TARGETS)].copy()
    loops = summary[summary["trial_id"].isin(LOOP_TRIALS)].copy()
    fusion = summary[summary["method"] == "magnetic_gyro_fusion"].set_index("trial_id")
    lines = [
        "# 初期直進補正を用いない磁気・ジャイロ融合比較",
        "",
        "## 比較方針",
        "",
        "原因切り分けを優先し、最初の2歩、3歩、6歩を正面へ揃える処理は用いていない。比較対象は次の二つだけである。",
        "",
        "| 方法 | 内容 |",
        "| --- | --- |",
        "| `simple (legacy)` | 従来の加速度ノルム、端末 `gyro_z` 積分、固定歩幅による軌跡 |",
        "| `magnetic/gyro fusion` | 重力方向ジャイロと傾き補償磁気方位を用いる SmartPDR 風軌跡。初期直進整列なし |",
        "",
        "この段階では磁気外乱の新しい棄却規則はまだ加えていない。現行の磁気融合が直進と周回へ与える影響を混乱なく評価し、次に導入する磁気外乱対策の根拠を得るためである。",
        "",
        "## 直進試行",
        "",
        "| 試行 | simple 終点誤差 [m] | magnetic/gyro fusion 終点誤差 [m] |",
        "| --- | ---: | ---: |",
    ]
    for trial_id in STRAIGHT_TARGETS:
        values = straight[straight["trial_id"] == trial_id].set_index("method")
        lines.append(
            f"| {trial_id} | {values.loc['simple_legacy', 'straight_endpoint_error_m']:.3f} | "
            f"{values.loc['magnetic_gyro_fusion', 'straight_endpoint_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 周回試行",
            "",
            "| 試行 | simple 閉路誤差 [m] | magnetic/gyro fusion 閉路誤差 [m] |",
            "| --- | ---: | ---: |",
        ]
    )
    for trial_id in sorted(LOOP_TRIALS):
        values = loops[loops["trial_id"] == trial_id].set_index("method")
        lines.append(
            f"| {trial_id} | {values.loc['simple_legacy', 'loop_closure_error_m']:.3f} | "
            f"{values.loc['magnetic_gyro_fusion', 'loop_closure_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 磁気融合の診断",
            "",
            "| 試行 | 磁気補正採用率 [%] | 磁場中央値 [uT] | 最大偏差 [uT] | 融合後方位変化 [deg] | ジャイロ方位変化 [deg] |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial_id in [*STRAIGHT_TARGETS, *sorted(LOOP_TRIALS)]:
        row = fusion.loc[trial_id]
        lines.append(
            f"| {trial_id} | {row['mag_acceptance_pct']:.1f} | {row['mag_norm_median_ut']:.2f} | "
            f"{row['mag_norm_max_deviation_ut']:.2f} | {row['heading_change_deg']:.2f} | "
            f"{row['gyro_heading_change_deg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 読み取り方",
            "",
            "- 直進誤差がシンプル版より大きい場合、磁気融合または端末方位と身体進行方向の差が直進表示を傾けている可能性がある。",
            "- 周回閉路誤差が磁気融合で小さくなる場合、磁気情報を完全に捨てることも適切ではない。",
            "- 磁気補正採用率が非常に高いまま直進が悪化する場合、次の一段は磁場強度変動やジャイロとの不整合を利用して磁気補正を限定する検証である。",
            "",
            "## 生成物",
            "",
            "- 各試行の比較図: `output_test/magnetic_fusion/<dataset>/<trial>/figures/trajectory_comparison.png`",
            "- 各試行の磁気診断図: `output_test/magnetic_fusion/<dataset>/<trial>/figures/magnetic_diagnostics.png`",
            "- 直進誤差: `output_test/magnetic_fusion/metrics/straight_endpoint_error.png`",
            "- 周回閉路誤差: `output_test/magnetic_fusion/metrics/loop_closure_error.png`",
            "- 磁気採用率と磁場偏差: `output_test/magnetic_fusion/metrics/magnetic_diagnostics.png`",
            "- 数値一覧: `output_test/magnetic_fusion/comparison_summary.csv`",
        ]
    )
    (output_root / "magnetic_fusion_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
