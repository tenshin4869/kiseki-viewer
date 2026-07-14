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
    parser = argparse.ArgumentParser(description="Plot endpoint density per A-table seat.")
    parser.add_argument("--manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_endpoint_density_by_seat")
    parser.add_argument("--grid-size", type=int, default=140)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    endpoints = pd.read_csv(args.manifest)
    endpoints = endpoints[endpoints["trajectory_name"].isin(TARGETS)].copy()

    summary_rows = []
    global_limits = _global_limits(endpoints)
    for label in sorted(TARGETS):
        group = endpoints[endpoints["trajectory_name"] == label].copy()
        if group.empty:
            continue
        mean_x = float(group["endpoint_x"].mean())
        mean_y = float(group["endpoint_y"].mean())
        target_x, target_y = TARGETS[label]
        summary_rows.append(
            {
                "trajectory_name": label,
                "count": int(len(group)),
                "target_x": target_x,
                "target_y": target_y,
                "mean_endpoint_x": mean_x,
                "mean_endpoint_y": mean_y,
                "mean_to_target_distance_m": float(np.hypot(mean_x - target_x, mean_y - target_y)),
            }
        )
        density = _density_grid(group, args.grid_size)
        _plot_2d(label, group, density, output_dir / f"{label}_endpoint_density_2d.png", global_limits)
        _plot_3d(label, group, density, output_dir / f"{label}_endpoint_density_3d.png", global_limits)

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "density_by_seat_summary.csv"
    summary.to_csv(summary_path, index=False)
    _plot_combined_2d(endpoints, output_dir / "A_all_seats_endpoint_density_2d_grid.png", global_limits)
    _plot_combined_3d(endpoints, output_dir / "A_all_seats_endpoint_density_3d_grid.png", global_limits, args.grid_size)
    all_density = _density_grid(
        endpoints.rename(columns={"endpoint_x": "x", "endpoint_y": "y"}),
        args.grid_size,
        x_col="x",
        y_col="y",
    )
    _plot_all_points_2d(endpoints, all_density, output_dir / "A_endpoint_density_all_points_2d.png", global_limits)
    _plot_all_points_3d(endpoints, all_density, output_dir / "A_endpoint_density_all_points_3d.png", global_limits)
    print(f"Summary: {summary_path}")
    print(f"Combined 2D: {output_dir / 'A_all_seats_endpoint_density_2d_grid.png'}")
    print(f"Combined 3D: {output_dir / 'A_all_seats_endpoint_density_3d_grid.png'}")
    print(f"All points 2D: {output_dir / 'A_endpoint_density_all_points_2d.png'}")
    print(f"All points 3D: {output_dir / 'A_endpoint_density_all_points_3d.png'}")
    for label in sorted(TARGETS):
        print(f"{label} 2D: {output_dir / f'{label}_endpoint_density_2d.png'}")
        print(f"{label} 3D: {output_dir / f'{label}_endpoint_density_3d.png'}")


def _density_grid(
    group: pd.DataFrame,
    grid_size: int,
    *,
    x_col: str = "endpoint_x",
    y_col: str = "endpoint_y",
) -> dict[str, np.ndarray]:
    x = group[x_col].to_numpy(dtype=float)
    y = group[y_col].to_numpy(dtype=float)
    pad_x = max(0.45, (x.max() - x.min()) * 0.25)
    pad_y = max(0.45, (y.max() - y.min()) * 0.25)
    x_min, x_max = x.min() - pad_x, x.max() + pad_x
    y_min, y_max = y.min() - pad_y, y.max() + pad_y
    xx, yy = np.mgrid[x_min:x_max:complex(grid_size), y_min:y_max:complex(grid_size)]
    coords = np.vstack([xx.ravel(), yy.ravel()])
    points = np.vstack([x, y])
    kde = gaussian_kde(points)
    zz = kde(coords).reshape(xx.shape)
    return {"xx": xx, "yy": yy, "zz": zz}


def _global_limits(endpoints: pd.DataFrame) -> tuple[float, float, float, float]:
    target_x = np.array([value[0] for value in TARGETS.values()])
    target_y = np.array([value[1] for value in TARGETS.values()])
    all_x = np.concatenate([endpoints["endpoint_x"].to_numpy(dtype=float), target_x])
    all_y = np.concatenate([endpoints["endpoint_y"].to_numpy(dtype=float), target_y])
    pad_x = max(0.55, (all_x.max() - all_x.min()) * 0.08)
    pad_y = max(0.55, (all_y.max() - all_y.min()) * 0.08)
    return all_x.min() - pad_x, all_x.max() + pad_x, all_y.min() - pad_y, all_y.max() + pad_y


def _plot_2d(
    label: str,
    group: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    target_x, target_y = TARGETS[label]
    color = COLORS[label]

    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    image = ax.contourf(
        density["xx"],
        density["yy"],
        density["zz"],
        levels=18,
        cmap="YlOrRd",
        alpha=0.70,
    )
    fig.colorbar(image, ax=ax, label="endpoint density")
    ax.scatter(
        group["endpoint_x"],
        group["endpoint_y"],
        s=28,
        color=color,
        alpha=0.60,
        edgecolors="white",
        linewidths=0.35,
        label="endpoints",
        zorder=5,
    )
    ax.scatter(
        [target_x],
        [target_y],
        marker="*",
        s=270,
        color="gold",
        edgecolor="black",
        linewidth=1.0,
        label="true seat",
        zorder=5,
    )
    _draw_table_a(ax)
    ax.set_title(f"{label}: endpoint density and spread")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    if limits:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_3d(
    label: str,
    group: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float] | None = None,
) -> None:
    target_x, target_y = TARGETS[label]
    color = COLORS[label]
    xx = density["xx"]
    yy = density["yy"]
    zz = density["zz"]
    z_floor = 0.0
    z_top = float(zz.max())

    fig = plt.figure(figsize=(8.4, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.88)
    ax.scatter(
        group["endpoint_x"],
        group["endpoint_y"],
        np.full(len(group), z_floor),
        s=18,
        color=color,
        alpha=0.55,
        depthshade=False,
        label="endpoints",
    )
    ax.scatter(
        [target_x],
        [target_y],
        [z_floor],
        marker="*",
        s=170,
        color="gold",
        edgecolor="black",
        depthshade=False,
        label="true seat",
    )
    ax.plot([target_x, target_x], [target_y, target_y], [z_floor, z_top], color="gold", linewidth=1.1)
    ax.set_title(f"{label}: endpoint density surface")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_zlabel("density")
    if limits:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
    ax.view_init(elev=32, azim=-55)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_combined_2d(
    endpoints: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, label in zip(axes.ravel(), sorted(TARGETS)):
        group = endpoints[endpoints["trajectory_name"] == label]
        density = _density_grid(group, 120)
        color = COLORS[label]
        target_x, target_y = TARGETS[label]
        ax.contourf(
            density["xx"],
            density["yy"],
            density["zz"],
            levels=16,
            cmap="YlOrRd",
            alpha=0.68,
        )
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=24,
            color=color,
            alpha=0.62,
            edgecolors="white",
            linewidths=0.25,
            zorder=5,
        )
        ax.scatter([target_x], [target_y], marker="*", s=230, color="gold", edgecolor="black", zorder=5)
        _draw_table_a(ax)
        ax.set_title(f"{label}  n={len(group)}")
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.22)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="tab:blue", markersize=7, label="endpoints"),
        plt.Line2D([0], [0], color="black", linewidth=3.0, label="table A"),
        plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="gold", markeredgecolor="black", markersize=13, label="true seat"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.98))
    fig.supxlabel("endpoint x [m]")
    fig.supylabel("endpoint y [m]")
    fig.suptitle("Endpoint density by A-seat: 2D view", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_3d(
    endpoints: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
    grid_size: int,
) -> None:
    fig = plt.figure(figsize=(13, 10))
    z_max = 0.0
    densities: dict[str, dict[str, np.ndarray]] = {}
    for label in sorted(TARGETS):
        group = endpoints[endpoints["trajectory_name"] == label]
        density = _density_grid(group, grid_size)
        densities[label] = density
        z_max = max(z_max, float(density["zz"].max()))

    for index, label in enumerate(sorted(TARGETS), start=1):
        group = endpoints[endpoints["trajectory_name"] == label]
        density = densities[label]
        target_x, target_y = TARGETS[label]
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
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            np.zeros(len(group)),
            s=14,
            color=COLORS[label],
            alpha=0.50,
            depthshade=False,
        )
        ax.scatter([target_x], [target_y], [0], marker="*", s=115, color="gold", edgecolor="black", depthshade=False)
        ax.set_title(label)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_zlim(0, z_max)
        ax.set_xlabel("x [m]", labelpad=2)
        ax.set_ylabel("y [m]", labelpad=2)
        ax.set_zlabel("density", labelpad=2)
        ax.view_init(elev=34, azim=-58)
    fig.suptitle("Endpoint density by A-seat: 3D density surfaces", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_all_points_2d(
    endpoints: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.contourf(density["xx"], density["yy"], density["zz"], levels=22, cmap="YlOrRd", alpha=0.75)
    fig.colorbar(image, ax=ax, label="endpoint density")
    for seat, group in endpoints.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=20,
            color=COLORS[str(seat)],
            alpha=0.50,
            edgecolors="none",
            label=str(seat),
            zorder=5,
        )
    _draw_all_targets(ax, endpoints)
    _draw_table_a(ax)
    ax.set_title("All A-seat endpoints: single density field")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_all_points_3d(
    endpoints: pd.DataFrame,
    density: dict[str, np.ndarray],
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy, zz = density["xx"], density["yy"], density["zz"]
    z_top = float(zz.max())
    ax.plot_surface(xx, yy, zz, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.90)
    for seat, group in endpoints.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            np.zeros(len(group)),
            s=15,
            color=COLORS[str(seat)],
            alpha=0.46,
            depthshade=False,
            label=str(seat),
        )
    for seat, (target_x, target_y) in TARGETS.items():
        ax.scatter([target_x], [target_y], [0], marker="*", s=130, color="gold", edgecolor="black", depthshade=False)
        ax.plot([target_x, target_x], [target_y, target_y], [0, z_top], color="gold", linewidth=0.8, alpha=0.7)
    ax.set_title("All A-seat endpoints: single 3D density surface")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_zlabel("density")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.view_init(elev=34, azim=-58)
    ax.legend(loc="upper left", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_all_targets(ax: plt.Axes, endpoints: pd.DataFrame) -> None:
    for seat, (target_x, target_y) in TARGETS.items():
        ax.scatter([target_x], [target_y], marker="*", s=210, color="gold", edgecolor="black", linewidth=0.9, zorder=5)
        ax.annotate(seat, (target_x, target_y), xytext=(5, 5), textcoords="offset points", fontsize=9)


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
