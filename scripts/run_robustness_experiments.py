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
from pdr_visualizer.io import find_trial_dirs, holding_position_from_trial
from pdr_visualizer.overlay import plot_all_trials
from pdr_visualizer.trial import run_trial


VARIANTS = {
    "no_alignment": {
        "label": "No alignment",
        "enabled": False,
        "step_count": 0,
        "assumption": "No known initial walking direction.",
    },
    "initial_2_steps": {
        "label": "First 2 steps as local forward",
        "enabled": True,
        "step_count": 2,
        "assumption": "The first 2 steps approximate one forward direction.",
    },
    "initial_3_steps": {
        "label": "First 3 steps as local forward",
        "enabled": True,
        "step_count": 3,
        "assumption": "The first 3 steps approximate one forward direction.",
    },
    "initial_6_steps_reference": {
        "label": "First 6 steps to known +y (current reference)",
        "enabled": True,
        "step_count": 6,
        "assumption": "The first 6 steps are known to follow the plotted +y axis.",
    },
}

STRAIGHT_BENCHMARKS = {
    "Data2/001_walk": np.array([0.0, 6.0]),
    "Data3/Walk_002": np.array([0.0, 6.0]),
}

LOOP_BENCHMARKS = {
    "Data2/LTurn_001",
    "Data2/RTurn_001",
    "Data3/LTurn_002",
    "Data3/RTurn_002",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate initial-heading assumptions without overwriting output/."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="output_test")
    args = parser.parse_args()

    base_config = load_config(args.config)
    output_root = Path(args.output_dir)
    trials = [
        str(path.relative_to(base_config["paths"]["raw_data_dir"]))
        for path in find_trial_dirs(base_config["paths"]["raw_data_dir"])
    ]

    rows: list[dict[str, object]] = []
    trajectories: dict[str, dict[str, pd.DataFrame]] = {}
    for variant, details in VARIANTS.items():
        config = deepcopy(base_config)
        config["paths"]["output_dir"] = str(output_root / "variants" / variant)
        config["heading"]["initial_alignment"]["enabled"] = bool(details["enabled"])
        if details["enabled"]:
            config["heading"]["initial_alignment"]["step_count"] = int(details["step_count"])

        trajectories[variant] = {}
        for trial_id in trials:
            paths = run_trial(trial_id, config)
            trajectory = pd.read_csv(paths["trajectory"])
            heading = pd.read_csv(paths["heading"])
            trajectories[variant][trial_id] = trajectory
            rows.append(_summarize_trial(variant, details, trial_id, trajectory, heading))
        plot_all_trials(config)

    summary = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_root / "robustness_summary.csv", index=False)
    _plot_trial_variant_comparisons(output_root, trajectories)
    _plot_straight_endpoint_comparison(output_root, summary)
    _plot_loop_closure_comparison(output_root, summary)
    stress = _plot_immediate_turn_stress_test(output_root, trajectories)
    stress.to_csv(output_root / "immediate_turn_stress_summary.csv", index=False)
    _write_report(output_root, summary, stress)
    print(f"summary: {output_root / 'robustness_summary.csv'}")
    print(f"report: {output_root / 'robustness_report.md'}")
    print(f"comparisons: {output_root / 'comparisons'}")


def _summarize_trial(
    variant: str,
    details: dict[str, object],
    trial_id: str,
    trajectory: pd.DataFrame,
    heading: pd.DataFrame,
) -> dict[str, object]:
    end = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
    first_leg_count = min(9, len(trajectory))
    first_leg_end = trajectory[["x", "y"]].iloc[first_leg_count - 1].to_numpy(dtype=float)
    alignment_deg = (
        float(np.degrees(heading["heading_alignment_offset_rad"].iloc[0]))
        if "heading_alignment_offset_rad" in heading
        else 0.0
    )
    expected = STRAIGHT_BENCHMARKS.get(trial_id)
    return {
        "variant": variant,
        "variant_label": details["label"],
        "assumption": details["assumption"],
        "trial_id": trial_id,
        "condition": holding_position_from_trial(trial_id),
        "step_count": len(trajectory),
        "distance_m": float(trajectory["step_length_m"].sum()),
        "end_x_m": end[0],
        "end_y_m": end[1],
        "endpoint_radius_m": float(np.linalg.norm(end)),
        "initial_9_steps_x_m": first_leg_end[0],
        "initial_9_steps_y_m": first_leg_end[1],
        "alignment_offset_deg": alignment_deg,
        "straight_target_error_m": (
            float(np.linalg.norm(end - expected)) if expected is not None else np.nan
        ),
        "loop_closure_error_m": (
            float(np.linalg.norm(end)) if trial_id in LOOP_BENCHMARKS else np.nan
        ),
    }


def _plot_trial_variant_comparisons(
    output_root: Path, trajectories: dict[str, dict[str, pd.DataFrame]]
) -> None:
    trial_ids = next(iter(trajectories.values())).keys()
    colors = {
        "no_alignment": "tab:gray",
        "initial_2_steps": "tab:orange",
        "initial_3_steps": "tab:green",
        "initial_6_steps_reference": "tab:blue",
    }
    for trial_id in trial_ids:
        fig, ax = plt.subplots(figsize=(8, 8))
        for variant, details in VARIANTS.items():
            trajectory = trajectories[variant][trial_id]
            x = [0.0, *trajectory["x"].to_list()]
            y = [0.0, *trajectory["y"].to_list()]
            ax.plot(x, y, marker="o", markersize=2.5, color=colors[variant], label=details["label"])
            ax.scatter([x[-1]], [y[-1]], marker="x", s=45, color=colors[variant])
        ax.scatter([0], [0], marker="s", s=48, color="black", label="start", zorder=4)
        ax.set_title(f"{trial_id}: initial-heading assumption comparison")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.axis("equal")
        ax.grid(True)
        ax.legend(fontsize="small")
        _save(fig, output_root / "comparisons" / trial_id / "alignment_variants.png")


def _plot_straight_endpoint_comparison(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(STRAIGHT_BENCHMARKS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for trial_id, group in selection.groupby("trial_id", sort=True):
        group = group.set_index("variant").loc[list(VARIANTS)]
        ax.plot(
            [VARIANTS[name]["label"] for name in VARIANTS],
            group["straight_target_error_m"],
            marker="o",
            label=trial_id,
        )
    ax.set_ylabel("Endpoint error to known straight target [m]")
    ax.set_title("Straight benchmark: effect of initial-heading assumptions")
    ax.grid(True, axis="y")
    ax.legend()
    ax.tick_params(axis="x", rotation=18)
    _save(fig, output_root / "comparisons" / "metrics" / "straight_endpoint_error.png")


def _plot_loop_closure_comparison(output_root: Path, summary: pd.DataFrame) -> None:
    selection = summary[summary["trial_id"].isin(LOOP_BENCHMARKS)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    for trial_id, group in selection.groupby("trial_id", sort=True):
        group = group.set_index("variant").loc[list(VARIANTS)]
        ax.plot(
            [VARIANTS[name]["label"] for name in VARIANTS],
            group["loop_closure_error_m"],
            marker="o",
            label=trial_id,
        )
    ax.set_ylabel("Closure error [m]")
    ax.set_title("Loop benchmark: global rotation does not repair shape")
    ax.grid(True, axis="y")
    ax.legend()
    ax.tick_params(axis="x", rotation=18)
    _save(fig, output_root / "comparisons" / "metrics" / "loop_closure_error.png")


def _plot_immediate_turn_stress_test(
    output_root: Path, trajectories: dict[str, dict[str, pd.DataFrame]]
) -> pd.DataFrame:
    source_trial = "Data2/LTurn_001"
    source = trajectories["initial_6_steps_reference"][source_trial]
    unwrapped = np.unwrap(source["heading_rad"].to_numpy(dtype=float))
    turn_indices = np.flatnonzero(np.abs(np.degrees(unwrapped)) >= 20.0)
    start_step = max(0, int(turn_indices[0]) - 1) if len(turn_indices) else 8
    segment = source.iloc[start_step:].reset_index(drop=True)
    lengths = segment["step_length_m"].to_numpy(dtype=float)
    headings = segment["heading_rad"].to_numpy(dtype=float)
    reference = _positions_from_headings(lengths, headings)
    tests = {
        "map_reference": ("Map-direction reference", headings, "black"),
        "initial_2_steps": (
            "First 2 steps forced to local forward",
            headings - _circular_mean(headings[:2]),
            "tab:orange",
        ),
        "initial_3_steps": (
            "First 3 steps forced to local forward",
            headings - _circular_mean(headings[:3]),
            "tab:green",
        ),
        "initial_6_steps": (
            "First 6 steps forced to local forward",
            headings - _circular_mean(headings[:6]),
            "tab:blue",
        ),
    }
    rows = []
    fig, ax = plt.subplots(figsize=(8, 8))
    for variant, (label, transformed, color) in tests.items():
        positions = _positions_from_headings(lengths, transformed)
        points = np.vstack([np.zeros(2), positions])
        endpoint_error = float(np.linalg.norm(positions[-1] - reference[-1]))
        rows.append(
            {
                "variant": variant,
                "label": label,
                "source_trial": source_trial,
                "source_start_step": start_step,
                "steps": len(lengths),
                "endpoint_error_to_map_reference_m": endpoint_error,
            }
        )
        ax.plot(points[:, 0], points[:, 1], marker="o", markersize=3, color=color, label=label)
        ax.scatter([points[-1, 0]], [points[-1, 1]], marker="x", s=48, color=color)
    ax.scatter([0], [0], marker="s", s=48, color="black", label="entry start", zorder=5)
    ax.set_title("Immediate-turn replay: risk of assuming initial forward steps")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend(fontsize="small")
    _save(fig, output_root / "comparisons" / "stress_test" / "immediate_turn_alignment.png")
    return pd.DataFrame(rows)


def _positions_from_headings(lengths: np.ndarray, headings: np.ndarray) -> np.ndarray:
    increments = np.column_stack((lengths * np.sin(headings), lengths * np.cos(headings)))
    return np.cumsum(increments, axis=0)


def _circular_mean(headings: np.ndarray) -> float:
    return float(np.angle(np.mean(np.exp(1j * headings))))


def _write_report(output_root: Path, summary: pd.DataFrame, stress: pd.DataFrame) -> None:
    straight = summary[summary["trial_id"].isin(STRAIGHT_BENCHMARKS)].copy()
    loops = summary[summary["trial_id"].isin(LOOP_BENCHMARKS)].copy()
    lines = [
        "# 初期方位の仮定を外した軌跡可視化の頑健性検証",
        "",
        "## 目的",
        "",
        "店舗入口から任意の向き・行動で歩き始める利用場面を想定し、`y` 軸正方向に進むことや最初の6歩が直進であることを前提にしない場合の軌跡可視化を比較する。",
        "",
        "今回の結果は既存の `output/` を変更せず、すべて `output_test/` に出力した。",
        "",
        "## 比較条件",
        "",
        "| 条件 | 仮定 |",
        "| --- | --- |",
    ]
    variant_descriptions = {
        "no_alignment": "入口方位も初期直進も仮定しない。",
        "initial_2_steps": "最初の2歩だけはおおむね前方へ進むとみなし、表示上の正面に合わせる。",
        "initial_3_steps": "最初の3歩だけはおおむね前方へ進むとみなし、表示上の正面に合わせる。",
        "initial_6_steps_reference": "現在実装と同じく、最初の6歩が既知の正面方向であるとみなす参照条件。",
    }
    for variant, details in VARIANTS.items():
        lines.append(f"| {details['label']} | {variant_descriptions[variant]} |")
    lines.extend(
        [
            "",
            "## 既知の直進試行での評価",
            "",
            "直進試行では便宜上、到達すべき位置を `(0 m, 6 m)` として終点誤差を測った。これは比較用の正解であり、未知の店舗入口で自動的に得られる情報ではない。",
            "",
            "| 条件 | Data2/001_walk 終点誤差 [m] | Data3/Walk_002 終点誤差 [m] |",
            "| --- | ---: | ---: |",
        ]
    )
    for variant, details in VARIANTS.items():
        values = straight[straight["variant"] == variant].set_index("trial_id")
        lines.append(
            f"| {details['label']} | {values.loc['Data2/001_walk', 'straight_target_error_m']:.3f} | "
            f"{values.loc['Data3/Walk_002', 'straight_target_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 周回軌跡の閉路誤差",
            "",
            "初期歩数による整列は、軌跡全体に同じ角度の回転を掛ける処理である。従って、図の向きは変わるが、曲がり角や閉じ方など軌跡自体の形状誤差は修復しない。実際に閉路誤差はすべての条件で同じだった。",
            "",
            "| 条件 | Data2/LTurn_001 [m] | Data2/RTurn_001 [m] | Data3/LTurn_002 [m] | Data3/RTurn_002 [m] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, details in VARIANTS.items():
        values = loops[loops["variant"] == variant].set_index("trial_id")
        lines.append(
            f"| {details['label']} | {values.loc['Data2/LTurn_001', 'loop_closure_error_m']:.3f} | "
            f"{values.loc['Data2/RTurn_001', 'loop_closure_error_m']:.3f} | "
            f"{values.loc['Data3/LTurn_002', 'loop_closure_error_m']:.3f} | "
            f"{values.loc['Data3/RTurn_002', 'loop_closure_error_m']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 入店直後に曲がる場合の再生ストレステスト",
            "",
            "`Data2/LTurn_001` の実測軌跡から最初の旋回が始まる区間以降を切り出し、その地点を新しい入口開始位置として再生した。現在の収録データに「入口直後から曲がる」実測試行がないため、新規計測の代替として仮定破れの影響を確認するテストである。",
            "",
            "| 条件 | 地図方向を保持した参照軌跡との終点差 [m] |",
            "| --- | ---: |",
        ]
    )
    for _, row in stress.iterrows():
        lines.append(f"| {row['label']} | {row['endpoint_error_to_map_reference_m']:.3f} |")
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "- `No alignment` は最も仮定が少なく、センサから得た相対的な軌跡形状をそのまま出す。ただし、入口の絶対方位が不明なままでは店舗図面上の向きに配置できない。",
            "- 最初の2歩または3歩を用いる方式は、利用者に短い自然な入店直進を依頼できる場合の妥協案になる。これは表示上のローカル正面を定めるだけで、店舗の絶対方位を復元する処理ではない。",
            "- 最初の6歩を用いる現在の条件は、最初が直進となる今回の試行では最も直進誤差が小さい。しかし、入店直後に商品棚へ曲がる利用者に対しては、その曲がりを正面方向として回転させる危険がある。",
            "- 入口から自由に歩き始めても店舗図面へ安定して重ねるには、入口座標と入口内向き方位を地図データとして与える、端末方位と歩行方向の差を較正する、通路制約を利用したマップマッチングを行う、などの外部拘束が必要になる。",
            "",
            "## 現段階の推奨",
            "",
            "頑健性を優先する本番方針は、軌跡推定では初期直進整列を必須にせず、店舗入口ごとに既知の開始座標と入店方位を設定して座標系へ配置することが望ましい。入口方位を登録できない実験段階では、`No alignment` を基準とし、許容できる場合のみ最初の2歩または3歩をローカル表示調整として比較する。",
            "",
            "## 生成物",
            "",
            "- 全試行の条件比較図: `output_test/comparisons/<dataset>/<trial>/alignment_variants.png`",
            "- 直進終点誤差図: `output_test/comparisons/metrics/straight_endpoint_error.png`",
            "- 周回閉路誤差図: `output_test/comparisons/metrics/loop_closure_error.png`",
            "- 入店直後旋回ストレステスト図: `output_test/comparisons/stress_test/immediate_turn_alignment.png`",
            "- 数値一覧: `output_test/robustness_summary.csv` および `output_test/immediate_turn_stress_summary.csv`",
            "- 条件別の軌跡出力: `output_test/variants/<variant>/...`",
        ]
    )
    (output_root / "robustness_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
