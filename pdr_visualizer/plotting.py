from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_acc_norm(acc_df: pd.DataFrame, steps_df: pd.DataFrame, output_path: str | Path, dpi: int, show_grid: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acc_df["t"], acc_df["acc_norm"], label="acc_norm", alpha=0.45)
    ax.plot(acc_df["t"], acc_df["acc_norm_smooth"], label="acc_norm_smooth")
    if not steps_df.empty:
        ax.scatter(steps_df["step_time"], steps_df["acc_norm_smooth"], label="steps", s=20, zorder=3)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration norm [m/s^2]")
    ax.legend()
    ax.grid(show_grid)
    _save(fig, output_path, dpi)


def plot_heading(heading_df: pd.DataFrame, gyro_axis: str, output_path: str | Path, dpi: int, show_grid: bool) -> None:
    axis_col = f"gyro_{gyro_axis}"
    corrected_col = f"{axis_col}_corrected"
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(heading_df["t"], heading_df[axis_col], label=axis_col, alpha=0.5)
    axes[0].plot(heading_df["t"], heading_df[corrected_col], label=corrected_col)
    axes[0].set_ylabel("Angular rate [rad/s]")
    axes[0].legend()
    axes[0].grid(show_grid)
    axes[1].plot(heading_df["t"], heading_df["heading_rad"], label="heading_rad")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Heading [rad]")
    axes[1].legend()
    axes[1].grid(show_grid)
    _save(fig, output_path, dpi)


def plot_trajectory(
    trajectory_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_trajectory(ax, trajectory_df, label=title)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    _save(fig, output_path, dpi)


def plot_overlay(
    trajectories: list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
    title: str,
    dpi: int,
    equal_axis: bool,
    show_grid: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    for trial_id, trajectory_df in trajectories:
        _draw_trajectory(ax, trajectory_df, label=f"{trial_id} (Data2)", mark_end=False)
    ax.set_title(title)
    _style_trajectory_axis(ax, equal_axis, show_grid)
    ax.legend(loc="best")
    _save(fig, output_path, dpi)


def _draw_trajectory(ax: plt.Axes, trajectory_df: pd.DataFrame, label: str, mark_end: bool = True) -> None:
    x = [0.0, *trajectory_df["x"].to_list()]
    y = [0.0, *trajectory_df["y"].to_list()]
    ax.plot(x, y, marker="o", markersize=3, label=label)
    ax.scatter([0.0], [0.0], marker="s", s=50, color="tab:green", label="start", zorder=4)
    if mark_end and len(x) > 1:
        ax.scatter([x[-1]], [y[-1]], marker="x", s=55, color="tab:red", label="end", zorder=4)


def _style_trajectory_axis(ax: plt.Axes, equal_axis: bool, show_grid: bool) -> None:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if equal_axis:
        ax.axis("equal")
    ax.grid(show_grid)


def _save(fig: plt.Figure, output_path: str | Path, dpi: int) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
