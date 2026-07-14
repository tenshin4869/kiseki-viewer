from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "pdr_visualizer_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pdr_visualizer_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGETS = {
    "A-1": (-2.0, 4.6),
    "A-2": (-2.7, 4.6),
    "A-3": (-2.0, 5.5),
    "A-4": (-2.7, 5.5),
}

COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze endpoint errors for A table seats.")
    parser.add_argument(
        "--manifest",
        default="output/a_trajectories_uncorrected/manifest.csv",
        help="Manifest produced by export_a_trajectories_uncorrected.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/a_endpoint_errors",
        help="Directory for error CSVs and figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    endpoints = pd.read_csv(args.manifest)
    endpoints = endpoints[endpoints["trajectory_name"].isin(TARGETS)].copy()
    endpoints["target_x"] = endpoints["trajectory_name"].map(lambda label: TARGETS[label][0])
    endpoints["target_y"] = endpoints["trajectory_name"].map(lambda label: TARGETS[label][1])
    endpoints["error_dx_m"] = endpoints["endpoint_x"] - endpoints["target_x"]
    endpoints["error_dy_m"] = endpoints["endpoint_y"] - endpoints["target_y"]
    endpoints["endpoint_error_m"] = np.hypot(endpoints["error_dx_m"], endpoints["error_dy_m"])

    details_path = output_dir / "endpoint_errors.csv"
    summary_path = output_dir / "endpoint_error_summary.csv"
    dataset_summary_path = output_dir / "endpoint_error_summary_by_dataset.csv"
    endpoints.to_csv(details_path, index=False)

    summary = _summary(endpoints, ["trajectory_name"])
    dataset_summary = _summary(endpoints, ["dataset", "trajectory_name"])
    summary.to_csv(summary_path, index=False)
    dataset_summary.to_csv(dataset_summary_path, index=False)

    _plot_endpoint_scatter(endpoints, figures_dir / "endpoint_scatter_with_targets.png")
    _plot_residual_scatter(endpoints, figures_dir / "endpoint_residual_scatter.png")
    _plot_error_boxplot(endpoints, figures_dir / "endpoint_error_boxplot_by_seat.png")
    _plot_error_by_dataset(endpoints, figures_dir / "endpoint_error_by_dataset.png")

    print(f"Endpoint errors: {details_path}")
    print(f"Summary: {summary_path}")
    print(f"Dataset summary: {dataset_summary_path}")
    print(f"Endpoint scatter: {figures_dir / 'endpoint_scatter_with_targets.png'}")
    print(f"Residual scatter: {figures_dir / 'endpoint_residual_scatter.png'}")
    print(f"Error boxplot: {figures_dir / 'endpoint_error_boxplot_by_seat.png'}")
    print(f"Error by dataset: {figures_dir / 'endpoint_error_by_dataset.png'}")
    print(summary.to_string(index=False))


def _summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        errors = group["endpoint_error_m"].to_numpy(dtype=float)
        row.update(
            {
                "count": int(len(group)),
                "target_x": float(group["target_x"].iloc[0]),
                "target_y": float(group["target_y"].iloc[0]),
                "mean_endpoint_x": float(group["endpoint_x"].mean()),
                "mean_endpoint_y": float(group["endpoint_y"].mean()),
                "mean_dx_m": float(group["error_dx_m"].mean()),
                "mean_dy_m": float(group["error_dy_m"].mean()),
                "mean_error_m": float(errors.mean()),
                "median_error_m": float(np.median(errors)),
                "std_error_m": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
                "rmse_m": float(np.sqrt(np.mean(errors**2))),
                "p95_error_m": float(np.percentile(errors, 95)),
                "max_error_m": float(errors.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_endpoint_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for label, group in df.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=24,
            alpha=0.55,
            color=COLORS[label],
            label=f"{label} endpoints",
        )
        target_x, target_y = TARGETS[label]
        ax.scatter(
            [target_x],
            [target_y],
            marker="*",
            s=240,
            color=COLORS[label],
            edgecolor="black",
            linewidth=0.9,
            label=f"{label} target",
            zorder=5,
        )
    ax.scatter([0.0], [0.0], marker="s", s=60, color="tab:green", label="start")
    ax.set_title("A-seat endpoint scatter vs target coordinates")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_residual_scatter(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    for label, group in df.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["error_dx_m"],
            group["error_dy_m"],
            s=24,
            alpha=0.55,
            color=COLORS[label],
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.scatter([0.0], [0.0], marker="*", s=220, color="gold", edgecolor="black", label="target")
    ax.set_title("Endpoint residuals from each seat target")
    ax.set_xlabel("endpoint_x - target_x [m]")
    ax.set_ylabel("endpoint_y - target_y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_error_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    labels = sorted(TARGETS)
    data = [df[df["trajectory_name"] == label]["endpoint_error_m"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_title("Endpoint distance error by A seat")
    ax.set_xlabel("seat")
    ax.set_ylabel("endpoint error [m]")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_error_by_dataset(df: pd.DataFrame, output_path: Path) -> None:
    summary = _summary(df, ["dataset", "trajectory_name"])
    labels = sorted(TARGETS)
    datasets = sorted(summary["dataset"].unique())
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, dataset in enumerate(datasets):
        subset = summary[summary["dataset"] == dataset].set_index("trajectory_name")
        values = [subset.loc[label, "mean_error_m"] for label in labels]
        ax.bar(x + (index - (len(datasets) - 1) / 2) * width, values, width, label=str(dataset))
    ax.set_title("Mean endpoint error by dataset")
    ax.set_xlabel("seat")
    ax.set_ylabel("mean endpoint error [m]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(title="dataset")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
