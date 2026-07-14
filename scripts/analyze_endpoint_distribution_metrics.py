from __future__ import annotations

import argparse
import itertools
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


COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}
CHI2_95_2D = 5.991464547107979


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze endpoint distribution metrics by seat.")
    parser.add_argument("--manifest", default="output/a_trajectories_first_stop_filtered/manifest.csv")
    parser.add_argument("--output-dir", default="output/a_endpoint_distribution_metrics_first_stop_filtered")
    parser.add_argument("--grid-size", type=int, default=220)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    endpoints = pd.read_csv(args.manifest)
    endpoints = endpoints[endpoints["trajectory_name"].isin(COLORS)].copy()
    endpoints = endpoints.rename(columns={"endpoint_x": "x", "endpoint_y": "y"})

    per_seat = _per_seat_metrics(endpoints)
    pairwise = _pairwise_metrics(endpoints, args.grid_size)
    classification = _nearest_centroid_classification(endpoints)

    per_seat.to_csv(output_dir / "endpoint_distribution_by_seat.csv", index=False)
    pairwise.to_csv(output_dir / "endpoint_distribution_pairwise.csv", index=False)
    classification["summary"].to_csv(output_dir / "endpoint_classification_summary.csv", index=False)
    classification["predictions"].to_csv(output_dir / "endpoint_classification_predictions.csv", index=False)

    _plot_confidence_ellipses(endpoints, per_seat, figures_dir / "endpoint_95_confidence_ellipses.png")
    _plot_pairwise_heatmaps(pairwise, figures_dir / "endpoint_pairwise_distance_overlap_heatmaps.png")

    print(f"Per-seat metrics: {output_dir / 'endpoint_distribution_by_seat.csv'}")
    print(f"Pairwise metrics: {output_dir / 'endpoint_distribution_pairwise.csv'}")
    print(f"Classification summary: {output_dir / 'endpoint_classification_summary.csv'}")
    print(f"Ellipse figure: {figures_dir / 'endpoint_95_confidence_ellipses.png'}")
    print(f"Pairwise heatmaps: {figures_dir / 'endpoint_pairwise_distance_overlap_heatmaps.png'}")


def _per_seat_metrics(endpoints: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seat, group in endpoints.groupby("trajectory_name", sort=True):
        xy = group[["x", "y"]].to_numpy(dtype=float)
        mean = xy.mean(axis=0)
        std = xy.std(axis=0, ddof=1)
        cov = np.cov(xy.T, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        principal = eigenvectors[:, 0]
        principal_angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
        semi_major = float(np.sqrt(CHI2_95_2D * eigenvalues[0]))
        semi_minor = float(np.sqrt(CHI2_95_2D * eigenvalues[1]))
        ellipse_area = float(np.pi * semi_major * semi_minor)
        inv_cov = np.linalg.pinv(cov)
        centered = xy - mean
        mahalanobis2 = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
        outlier_rate = float(np.mean(mahalanobis2 > CHI2_95_2D))
        rows.append(
            {
                "seat": seat,
                "count": int(len(group)),
                "mean_x": float(mean[0]),
                "mean_y": float(mean[1]),
                "std_x": float(std[0]),
                "std_y": float(std[1]),
                "cov_xx": float(cov[0, 0]),
                "cov_xy": float(cov[0, 1]),
                "cov_yy": float(cov[1, 1]),
                "principal_axis_angle_deg": principal_angle_deg,
                "principal_variance": float(eigenvalues[0]),
                "minor_variance": float(eigenvalues[1]),
                "axis_variance_ratio": float(eigenvalues[0] / eigenvalues[1]) if eigenvalues[1] > 0 else np.inf,
                "ellipse95_semi_major_m": semi_major,
                "ellipse95_semi_minor_m": semi_minor,
                "ellipse95_area_m2": ellipse_area,
                "outlier_rate_outside_ellipse95": outlier_rate,
            }
        )
    return pd.DataFrame(rows)


def _pairwise_metrics(endpoints: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    densities, cell_area = _normalized_kde_grids(endpoints, grid_size)
    rows = []
    grouped = {seat: group[["x", "y"]].to_numpy(dtype=float) for seat, group in endpoints.groupby("trajectory_name")}
    means = {seat: xy.mean(axis=0) for seat, xy in grouped.items()}
    covs = {seat: np.cov(xy.T, ddof=1) for seat, xy in grouped.items()}

    for seat_a, seat_b in itertools.combinations(sorted(grouped), 2):
        mean_distance = float(np.linalg.norm(means[seat_a] - means[seat_b]))
        spread_a = float(np.sqrt(np.trace(covs[seat_a])))
        spread_b = float(np.sqrt(np.trace(covs[seat_b])))
        pooled_spread = float(np.sqrt((np.trace(covs[seat_a]) + np.trace(covs[seat_b])) / 2.0))
        overlap = float(np.minimum(densities[seat_a], densities[seat_b]).sum() * cell_area)
        rows.append(
            {
                "seat_a": seat_a,
                "seat_b": seat_b,
                "mean_distance_m": mean_distance,
                "spread_a_m": spread_a,
                "spread_b_m": spread_b,
                "pooled_spread_m": pooled_spread,
                "mean_distance_div_pooled_spread": mean_distance / pooled_spread if pooled_spread > 0 else np.inf,
                "kde_overlap_coefficient": overlap,
                "relationship": _seat_relationship(seat_a, seat_b),
            }
        )
    return pd.DataFrame(rows)


def _normalized_kde_grids(endpoints: pd.DataFrame, grid_size: int) -> tuple[dict[str, np.ndarray], float]:
    x = endpoints["x"].to_numpy(dtype=float)
    y = endpoints["y"].to_numpy(dtype=float)
    pad_x = max(0.6, (x.max() - x.min()) * 0.08)
    pad_y = max(0.6, (y.max() - y.min()) * 0.08)
    x_grid = np.linspace(x.min() - pad_x, x.max() + pad_x, grid_size)
    y_grid = np.linspace(y.min() - pad_y, y.max() + pad_y, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid, indexing="xy")
    coords = np.vstack([xx.ravel(), yy.ravel()])
    cell_area = float((x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0]))
    densities: dict[str, np.ndarray] = {}
    for seat, group in endpoints.groupby("trajectory_name", sort=True):
        xy = group[["x", "y"]].to_numpy(dtype=float)
        kde = gaussian_kde(xy.T)
        density = kde(coords).reshape(xx.shape)
        density = density / (density.sum() * cell_area)
        densities[seat] = density
    return densities, cell_area


def _seat_relationship(seat_a: str, seat_b: str) -> str:
    index_a = int(seat_a.split("-")[1])
    index_b = int(seat_b.split("-")[1])
    pair = {index_a, index_b}
    if pair in ({1, 2}, {1, 3}, {2, 4}, {3, 4}):
        return "adjacent"
    return "diagonal"


def _nearest_centroid_classification(endpoints: pd.DataFrame) -> dict[str, pd.DataFrame]:
    predictions = []
    for index, row in endpoints.iterrows():
        train = endpoints.drop(index=index)
        centroids = train.groupby("trajectory_name")[["x", "y"]].mean()
        point = row[["x", "y"]].to_numpy(dtype=float)
        distances = {
            seat: float(np.linalg.norm(point - centroid.to_numpy(dtype=float)))
            for seat, centroid in centroids.iterrows()
        }
        predicted = min(distances, key=distances.get)
        predictions.append(
            {
                "trial_id": row["trial_id"],
                "true_label": row["trajectory_name"],
                "predicted_label": predicted,
                "is_correct": bool(predicted == row["trajectory_name"]),
                "nearest_centroid_distance_m": distances[predicted],
            }
        )
    predictions_df = pd.DataFrame(predictions)
    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "count": int(len(predictions_df)),
                "accuracy": float(predictions_df["is_correct"].mean()),
            }
        ]
    )
    per_seat = (
        predictions_df.groupby("true_label")
        .agg(count=("trial_id", "size"), accuracy=("is_correct", "mean"))
        .reset_index()
        .rename(columns={"true_label": "scope"})
    )
    summary = pd.concat([overall, per_seat], ignore_index=True)
    return {"summary": summary, "predictions": predictions_df}


def _plot_confidence_ellipses(endpoints: pd.DataFrame, metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for seat, group in endpoints.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["x"],
            group["y"],
            s=26,
            color=COLORS[seat],
            alpha=0.52,
            edgecolors="white",
            linewidths=0.25,
            label=seat,
        )
        row = metrics[metrics["seat"] == seat].iloc[0]
        _draw_ellipse(
            ax,
            center=(row["mean_x"], row["mean_y"]),
            semi_major=row["ellipse95_semi_major_m"],
            semi_minor=row["ellipse95_semi_minor_m"],
            angle_deg=row["principal_axis_angle_deg"],
            color=COLORS[seat],
        )
        ax.scatter([row["mean_x"]], [row["mean_y"]], marker="x", s=75, color=COLORS[seat], linewidth=2.0)
    ax.set_title("Endpoint distributions with 95% confidence ellipses")
    ax.set_xlabel("endpoint x [m]")
    ax.set_ylabel("endpoint y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_ellipse(
    ax: plt.Axes,
    center: tuple[float, float],
    semi_major: float,
    semi_minor: float,
    angle_deg: float,
    color: str,
) -> None:
    t = np.linspace(0, 2 * np.pi, 220)
    ellipse = np.vstack([semi_major * np.cos(t), semi_minor * np.sin(t)])
    theta = np.radians(angle_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = rotation @ ellipse
    ax.plot(rotated[0] + center[0], rotated[1] + center[1], color=color, linewidth=2.0, alpha=0.85)


def _plot_pairwise_heatmaps(pairwise: pd.DataFrame, output_path: Path) -> None:
    seats = sorted(set(pairwise["seat_a"]).union(pairwise["seat_b"]))
    distance = pd.DataFrame(np.nan, index=seats, columns=seats)
    overlap = pd.DataFrame(np.nan, index=seats, columns=seats)
    for _, row in pairwise.iterrows():
        a, b = row["seat_a"], row["seat_b"]
        distance.loc[a, b] = distance.loc[b, a] = row["mean_distance_m"]
        overlap.loc[a, b] = overlap.loc[b, a] = row["kde_overlap_coefficient"]
    for seat in seats:
        distance.loc[seat, seat] = 0.0
        overlap.loc[seat, seat] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, matrix, title, cmap in [
        (axes[0], distance, "Mean distance [m]", "Blues"),
        (axes[1], overlap, "KDE overlap coefficient", "Reds"),
    ]:
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap=cmap)
        ax.set_xticks(range(len(seats)), seats)
        ax.set_yticks(range(len(seats)), seats)
        ax.set_title(title)
        for i in range(len(seats)):
            for j in range(len(seats)):
                ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
