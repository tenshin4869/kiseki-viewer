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
from scipy.stats import gaussian_kde


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
TABLE_A = {
    "left": -3.3,
    "bottom": 4.6,
    "width": 1.5,
    "height": 0.9,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot density of seat-area segment representative points.")
    parser.add_argument("--segments", default="output/a_segments_direct_slowdown_only/segments.csv")
    parser.add_argument("--output-dir", default="output/a_representative_density_by_seat")
    parser.add_argument("--grid-size", type=int, default=140)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = pd.read_csv(args.segments)
    points = segments[segments["classification"] == "seat_area"].copy()
    points = points[points["destination_label"].isin(TARGETS)].copy()
    points = points.rename(
        columns={
            "destination_label": "seat",
            "representative_x": "x",
            "representative_y": "y",
        }
    )

    summary_rows = []
    limits = _global_limits(points)
    densities: dict[str, dict[str, np.ndarray]] = {}
    for seat in sorted(TARGETS):
        group = points[points["seat"] == seat].copy()
        if group.empty:
            continue
        density = _density_grid(group, args.grid_size)
        densities[seat] = density
        mean_x = float(group["x"].mean())
        mean_y = float(group["y"].mean())
        target_x, target_y = TARGETS[seat]
        summary_rows.append(
            {
                "seat": seat,
                "representative_count": int(len(group)),
                "target_x": target_x,
                "target_y": target_y,
                "mean_representative_x": mean_x,
                "mean_representative_y": mean_y,
                "mean_to_target_distance_m": float(np.hypot(mean_x - target_x, mean_y - target_y)),
            }
        )
        _plot_2d(seat, group, density, output_dir / f"{seat}_representative_density_2d.png", limits)
        _plot_3d(seat, group, density, output_dir / f"{seat}_representative_density_3d.png", limits)

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "representative_density_by_seat_summary.csv"
    points_path = output_dir / "seat_area_representative_points.csv"
    summary.to_csv(summary_path, index=False)
    points.to_csv(points_path, index=False)
    _plot_combined_2d(points, output_dir / "A_all_seats_representative_density_2d_grid.png", limits)
    _plot_combined_3d(points, densities, output_dir / "A_all_seats_representative_density_3d_grid.png", limits)
    all_density = _density_grid(points, args.grid_size)
    _plot_all_points_2d(points, all_density, output_dir / "A_representative_density_all_points_2d.png", limits)
    _plot_all_points_3d(points, all_density, output_dir / "A_representative_density_all_points_3d.png", limits)

    print(f"Representative points: {points_path}")
    print(f"Summary: {summary_path}")
    print(f"Combined 2D: {output_dir / 'A_all_seats_representative_density_2d_grid.png'}")
    print(f"Combined 3D: {output_dir / 'A_all_seats_representative_density_3d_grid.png'}")
    print(f"All points 2D: {output_dir / 'A_representative_density_all_points_2d.png'}")
    print(f"All points 3D: {output_dir / 'A_representative_density_all_points_3d.png'}")
    for seat in sorted(TARGETS):
        print(f"{seat} 2D: {output_dir / f'{seat}_representative_density_2d.png'}")
        print(f"{seat} 3D: {output_dir / f'{seat}_representative_density_3d.png'}")


def _density_grid(group: pd.DataFrame, grid_size: int) -> dict[str, np.ndarray]:
    x = group["x"].to_numpy(dtype=float)
    y = group["y"].to_numpy(dtype=float)
    pad_x = max(0.45, (x.max() - x.min()) * 0.25)
    pad_y = max(0.45, (y.max() - y.min()) * 0.25)
    xx, yy = np.mgrid[
        x.min() - pad_x : x.max() + pad_x : complex(grid_size),
        y.min() - pad_y : y.max() + pad_y : complex(grid_size),
    ]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(np.vstack([x, y]))
    zz = kde(grid).reshape(xx.shape)
    return {"xx": xx, "yy": yy, "zz": zz}


def _global_limits(points: pd.DataFrame) -> tuple[float, float, float, float]:
    target_x = np.array([value[0] for value in TARGETS.values()])
    target_y = np.array([value[1] for value in TARGETS.values()])
    all_x = np.concatenate([points["x"].to_numpy(dtype=float), target_x])
    all_y = np.concatenate([points["y"].to_numpy(dtype=float), target_y])
    pad_x = max(0.55, (all_x.max() - all_x.min()) * 0.08)
    pad_y = max(0.55, (all_y.max() - all_y.min()) * 0.08)
    return all_x.min() - pad_x, all_x.max() + pad_x, all_y.min() - pad_y, all_y.max() + pad_y


def _plot_2d(
    seat: str,
    group: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    ax.contourf(density["xx"], density["yy"], density["zz"], levels=18, cmap="YlOrRd", alpha=0.70)
    _draw_points(ax, seat, group)
    _draw_table_a(ax)
    ax.set_title(f"{seat}: seat-area representative density")
    ax.set_xlabel("representative x [m]")
    ax.set_ylabel("representative y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_3d(
    seat: str,
    group: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy, zz = density["xx"], density["yy"], density["zz"]
    ax.plot_surface(xx, yy, zz, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.88)
    _draw_points_3d(ax, seat, group, float(zz.max()))
    ax.set_title(f"{seat}: representative density surface")
    ax.set_xlabel("representative x [m]")
    ax.set_ylabel("representative y [m]")
    ax.set_zlabel("density")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.view_init(elev=32, azim=-55)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_combined_2d(points: pd.DataFrame, output_path: Path, limits: tuple[float, float, float, float]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, seat in zip(axes.ravel(), sorted(TARGETS)):
        group = points[points["seat"] == seat]
        density = _density_grid(group, 120)
        ax.contourf(density["xx"], density["yy"], density["zz"], levels=16, cmap="YlOrRd", alpha=0.68)
        _draw_points(ax, seat, group, legend=False)
        _draw_table_a(ax)
        ax.set_title(f"{seat}  n={len(group)}")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue", markersize=7, label="representatives"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="true seat"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("representative x [m]")
    fig.supylabel("representative y [m]")
    fig.suptitle("Seat-area representative density by A-seat: 2D view", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_3d(
    points: pd.DataFrame,
    densities: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig = plt.figure(figsize=(13, 10))
    z_max = max(float(density["zz"].max()) for density in densities.values())
    for index, seat in enumerate(sorted(TARGETS), start=1):
        group = points[points["seat"] == seat]
        density = densities[seat]
        ax = fig.add_subplot(2, 2, index, projection="3d")
        ax.plot_surface(
            density["xx"],
            density["yy"],
            density["zz"],
            cmap="YlOrRd",
            linewidth=0,
            antialiased=True,
            alpha=0.88,
        )
        _draw_points_3d(ax, seat, group, z_max)
        ax.set_title(seat)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(0, z_max)
        ax.set_xlabel("x [m]", labelpad=2)
        ax.set_ylabel("y [m]", labelpad=2)
        ax.set_zlabel("density", labelpad=2)
        ax.view_init(elev=34, azim=-58)
    fig.suptitle("Seat-area representative density by A-seat: 3D surfaces", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_all_points_2d(
    points: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.contourf(density["xx"], density["yy"], density["zz"], levels=22, cmap="YlOrRd", alpha=0.75)
    fig.colorbar(image, ax=ax, label="representative density")
    for seat, group in points.groupby("seat", sort=True):
        ax.scatter(
            group["x"],
            group["y"],
            s=18,
            color=COLORS[seat],
            alpha=0.48,
            edgecolors="none",
            label=seat,
            zorder=5,
        )
    _draw_all_targets(ax, points)
    _draw_table_a(ax)
    ax.set_title("All A-seat representative points: single density field")
    ax.set_xlabel("representative x [m]")
    ax.set_ylabel("representative y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_all_points_3d(
    points: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy, zz = density["xx"], density["yy"], density["zz"]
    z_top = float(zz.max())
    ax.plot_surface(xx, yy, zz, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.90)
    for seat, group in points.groupby("seat", sort=True):
        ax.scatter(
            group["x"],
            group["y"],
            np.zeros(len(group)),
            s=14,
            color=COLORS[seat],
            alpha=0.42,
            depthshade=False,
            label=seat,
        )
    for seat, (target_x, target_y) in TARGETS.items():
        ax.scatter([target_x], [target_y], [0], marker="*", s=130, color="gold", edgecolor="black", depthshade=False)
        ax.plot([target_x, target_x], [target_y, target_y], [0, z_top], color="gold", linewidth=0.8, alpha=0.7)
    ax.set_title("All A-seat representative points: single 3D density surface")
    ax.set_xlabel("representative x [m]")
    ax.set_ylabel("representative y [m]")
    ax.set_zlabel("density")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.view_init(elev=34, azim=-58)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_all_targets(ax: plt.Axes, points: pd.DataFrame) -> None:
    for seat, (target_x, target_y) in TARGETS.items():
        ax.scatter([target_x], [target_y], marker="*", s=210, color="gold", edgecolor="black", linewidth=0.9, zorder=5)
        ax.annotate(seat, (target_x, target_y), xytext=(5, 5), textcoords="offset points", fontsize=9)


def _draw_points(ax: plt.Axes, seat: str, group: pd.DataFrame, legend: bool = True) -> None:
    target_x, target_y = TARGETS[seat]
    ax.scatter(group["x"], group["y"], s=26, color=COLORS[seat], alpha=0.58, edgecolors="white", linewidths=0.3, label="representatives" if legend else None, zorder=5)
    ax.scatter([target_x], [target_y], marker="*", s=250, color="gold", edgecolor="black", linewidth=0.9, label="true seat" if legend else None, zorder=5)


def _draw_points_3d(ax: plt.Axes, seat: str, group: pd.DataFrame, z_top: float) -> None:
    target_x, target_y = TARGETS[seat]
    ax.scatter(group["x"], group["y"], np.zeros(len(group)), s=16, color=COLORS[seat], alpha=0.55, depthshade=False, label="representatives")
    ax.scatter([target_x], [target_y], [0], marker="*", s=140, color="gold", edgecolor="black", depthshade=False, label="true seat")
    ax.plot([target_x, target_x], [target_y, target_y], [0, z_top], color="gold", linewidth=1.0)


def _draw_table_a(ax: plt.Axes) -> None:
    rectangle = plt.Rectangle(
        (TABLE_A["left"], TABLE_A["bottom"]),
        TABLE_A["width"],
        TABLE_A["height"],
        fill=False,
        edgecolor="black",
        linewidth=3.0,
        zorder=4,
    )
    ax.add_patch(rectangle)


if __name__ == "__main__":
    main()
