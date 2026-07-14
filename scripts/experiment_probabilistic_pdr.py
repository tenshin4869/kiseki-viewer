from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimental probabilistic PDR visualization using Monte Carlo particles."
    )
    parser.add_argument("--manifest", default="output/a_trajectories_uncorrected/manifest.csv")
    parser.add_argument("--output-dir", default="output/probabilistic_pdr_experiment")
    parser.add_argument("--particles-per-trial", type=int, default=300)
    parser.add_argument("--step-length-sigma-m", type=float, default=0.05)
    parser.add_argument("--step-length-bias-sigma", type=float, default=0.04)
    parser.add_argument("--heading-step-sigma-deg", type=float, default=5.0)
    parser.add_argument("--heading-bias-sigma-deg", type=float, default=3.0)
    parser.add_argument("--last-progress-ratio", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)

    endpoint_frames = []
    late_position_frames = []
    sample_path_frames = []
    summary_rows: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        trajectory = pd.read_csv(row["trajectory_csv"])
        if trajectory.empty:
            continue
        simulated = _simulate_particles(
            trajectory,
            particles=args.particles_per_trial,
            step_length_sigma_m=args.step_length_sigma_m,
            step_length_bias_sigma=args.step_length_bias_sigma,
            heading_step_sigma_rad=np.deg2rad(args.heading_step_sigma_deg),
            heading_bias_sigma_rad=np.deg2rad(args.heading_bias_sigma_deg),
            rng=rng,
        )
        seat = str(row["trajectory_name"])
        trial_id = str(row["trial_id"])
        endpoints = simulated[:, -1, :]
        endpoint_frames.append(
            pd.DataFrame(
                {
                    "trial_id": trial_id,
                    "seat": seat,
                    "particle_id": np.arange(len(endpoints)),
                    "x": endpoints[:, 0],
                    "y": endpoints[:, 1],
                }
            )
        )

        start_index = max(0, int(np.floor((simulated.shape[1] - 1) * args.last_progress_ratio)))
        late = simulated[:, start_index:, :].reshape(-1, 2)
        late_position_frames.append(
            pd.DataFrame(
                {
                    "trial_id": trial_id,
                    "seat": seat,
                    "x": late[:, 0],
                    "y": late[:, 1],
                }
            )
        )

        sample_count = min(3, simulated.shape[0])
        for sample_id in range(sample_count):
            path = simulated[sample_id]
            sample_path_frames.append(
                pd.DataFrame(
                    {
                        "trial_id": trial_id,
                        "seat": seat,
                        "sample_id": sample_id,
                        "step": np.arange(len(path)),
                        "x": path[:, 0],
                        "y": path[:, 1],
                    }
                )
            )

        summary_rows.append(
            {
                "trial_id": trial_id,
                "seat": seat,
                "source_steps": int(len(trajectory)),
                "particles": int(args.particles_per_trial),
                "deterministic_endpoint_x": float(trajectory["x"].iloc[-1]),
                "deterministic_endpoint_y": float(trajectory["y"].iloc[-1]),
                "particle_endpoint_mean_x": float(endpoints[:, 0].mean()),
                "particle_endpoint_mean_y": float(endpoints[:, 1].mean()),
                "particle_endpoint_std_x": float(endpoints[:, 0].std(ddof=1)),
                "particle_endpoint_std_y": float(endpoints[:, 1].std(ddof=1)),
            }
        )

    endpoints_df = pd.concat(endpoint_frames, ignore_index=True)
    late_positions_df = pd.concat(late_position_frames, ignore_index=True)
    sample_paths_df = pd.concat(sample_path_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    endpoints_path = output_dir / "particle_endpoints.csv"
    late_positions_path = output_dir / "late_particle_positions.csv"
    sample_paths_path = output_dir / "sample_particle_paths.csv"
    summary_path = output_dir / "probabilistic_pdr_summary.csv"
    endpoints_df.to_csv(endpoints_path, index=False)
    late_positions_df.to_csv(late_positions_path, index=False)
    sample_paths_df.to_csv(sample_paths_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    limits = _limits([endpoints_df, late_positions_df])
    _plot_endpoint_density(endpoints_df, manifest, output_dir / "particle_endpoint_density_2d.png", limits)
    _plot_endpoint_density_3d(endpoints_df, output_dir / "particle_endpoint_density_3d.png", limits)
    _plot_late_position_density(
        late_positions_df,
        manifest,
        output_dir / "late_position_candidate_region_density_2d.png",
        limits,
    )
    _plot_by_seat_endpoint_grid(
        endpoints_df,
        output_dir / "particle_endpoint_density_by_seat_2x2.png",
        limits,
    )
    _plot_sample_paths(sample_paths_df, manifest, output_dir / "sample_probabilistic_paths.png", limits)

    parameters = pd.DataFrame(
        [
            {
                "particles_per_trial": args.particles_per_trial,
                "step_length_sigma_m": args.step_length_sigma_m,
                "step_length_bias_sigma": args.step_length_bias_sigma,
                "heading_step_sigma_deg": args.heading_step_sigma_deg,
                "heading_bias_sigma_deg": args.heading_bias_sigma_deg,
                "last_progress_ratio": args.last_progress_ratio,
                "seed": args.seed,
            }
        ]
    )
    parameters.to_csv(output_dir / "probabilistic_pdr_parameters.csv", index=False)

    print(f"Particle endpoints: {endpoints_path}")
    print(f"Late particle positions: {late_positions_path}")
    print(f"Summary: {summary_path}")
    print(f"Endpoint density 2D: {output_dir / 'particle_endpoint_density_2d.png'}")
    print(f"Endpoint density 3D: {output_dir / 'particle_endpoint_density_3d.png'}")
    print(f"Late candidate region density: {output_dir / 'late_position_candidate_region_density_2d.png'}")
    print(f"By-seat endpoint density: {output_dir / 'particle_endpoint_density_by_seat_2x2.png'}")
    print(f"Sample probabilistic paths: {output_dir / 'sample_probabilistic_paths.png'}")


def _simulate_particles(
    trajectory: pd.DataFrame,
    *,
    particles: int,
    step_length_sigma_m: float,
    step_length_bias_sigma: float,
    heading_step_sigma_rad: float,
    heading_bias_sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    base_lengths = trajectory["step_length_m"].to_numpy(dtype=float)
    base_headings = trajectory["heading_rad"].to_numpy(dtype=float)
    n_steps = len(base_lengths)
    length_bias = rng.normal(0.0, step_length_bias_sigma, size=(particles, 1))
    lengths = base_lengths[None, :] * (1.0 + length_bias)
    lengths += rng.normal(0.0, step_length_sigma_m, size=(particles, n_steps))
    lengths = np.clip(lengths, 0.05, None)
    heading_bias = rng.normal(0.0, heading_bias_sigma_rad, size=(particles, 1))
    headings = base_headings[None, :] + heading_bias
    headings += rng.normal(0.0, heading_step_sigma_rad, size=(particles, n_steps))
    dx = lengths * np.sin(headings)
    dy = lengths * np.cos(headings)
    return np.stack([np.cumsum(dx, axis=1), np.cumsum(dy, axis=1)], axis=2)


def _limits(frames: list[pd.DataFrame]) -> tuple[float, float, float, float]:
    x = np.concatenate([frame["x"].to_numpy(dtype=float) for frame in frames])
    y = np.concatenate([frame["y"].to_numpy(dtype=float) for frame in frames])
    pad_x = max(0.7, (x.max() - x.min()) * 0.08)
    pad_y = max(0.7, (y.max() - y.min()) * 0.08)
    return x.min() - pad_x, x.max() + pad_x, y.min() - pad_y, y.max() + pad_y


def _plot_endpoint_density(
    endpoints: pd.DataFrame,
    manifest: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    hist = ax.hist2d(endpoints["x"], endpoints["y"], bins=90, cmap="YlOrRd", range=[[limits[0], limits[1]], [limits[2], limits[3]]])
    fig.colorbar(hist[3], ax=ax, label="particle endpoint count")
    for seat, group in manifest.groupby("trajectory_name", sort=True):
        ax.scatter(
            group["endpoint_x"],
            group["endpoint_y"],
            s=16,
            color=COLORS.get(str(seat), "tab:gray"),
            alpha=0.48,
            edgecolors="none",
        )
    ax.set_title("Probabilistic PDR endpoint candidate density")
    _style_xy(ax, limits)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_endpoint_density_3d(
    endpoints: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    sample = endpoints.sample(min(len(endpoints), 60000), random_state=42)
    x_min, x_max, y_min, y_max = limits
    xx, yy = np.mgrid[x_min:x_max:130j, y_min:y_max:130j]
    kde = gaussian_kde(sample[["x", "y"]].to_numpy(dtype=float).T)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap="YlOrRd", linewidth=0, antialiased=True, alpha=0.90)
    ax.set_title("Probabilistic PDR endpoint density surface")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("density")
    ax.view_init(elev=34, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_late_position_density(
    late_positions: pd.DataFrame,
    manifest: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sample = late_positions.sample(min(len(late_positions), 250000), random_state=1)
    hist = ax.hist2d(sample["x"], sample["y"], bins=100, cmap="YlOrRd", range=[[limits[0], limits[1]], [limits[2], limits[3]]])
    fig.colorbar(hist[3], ax=ax, label="late particle-position count")
    ax.scatter(
        manifest["endpoint_x"],
        manifest["endpoint_y"],
        s=9,
        color="black",
        alpha=0.25,
        edgecolors="none",
    )
    ax.set_title("Candidate stay region density from late-step particles")
    _style_xy(ax, limits)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_by_seat_endpoint_grid(
    endpoints: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (seat, group) in zip(axes.ravel(), endpoints.groupby("seat", sort=True)):
        ax.hist2d(group["x"], group["y"], bins=55, cmap="YlOrRd", range=[[limits[0], limits[1]], [limits[2], limits[3]]])
        ax.set_title(f"{seat} particle endpoints")
        _style_xy(ax, limits)
    fig.suptitle("Probabilistic endpoint density by recorded seat label", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_paths(
    sample_paths: pd.DataFrame,
    manifest: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7.5))
    limited = sample_paths.groupby("seat", sort=True).head(1600)
    for (seat, trial_id, sample_id), path in limited.groupby(["seat", "trial_id", "sample_id"], sort=False):
        ax.plot(path["x"], path["y"], color=COLORS.get(str(seat), "tab:gray"), alpha=0.08, linewidth=0.7)
    ax.scatter(manifest["endpoint_x"], manifest["endpoint_y"], s=8, color="black", alpha=0.30, label="deterministic endpoints")
    ax.scatter([0.0], [0.0], marker="s", s=55, color="tab:green", label="start")
    ax.set_title("Sample probabilistic PDR paths")
    _style_xy(ax, limits)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _style_xy(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


if __name__ == "__main__":
    main()
