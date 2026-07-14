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


SEAT_COLORS = {
    "A-1": "tab:blue",
    "A-2": "tab:orange",
    "A-3": "tab:green",
    "A-4": "tab:red",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate straight-walk heading uncertainty through A-table PDR trajectories. "
            "The nominal trajectory is preserved; this creates plausible heading-drift variants."
        )
    )
    parser.add_argument(
        "--correction-summary",
        default="output/straight_walk_drift_analysis/trial_drift_summary.csv",
    )
    parser.add_argument(
        "--trajectory-manifest",
        default="output/a_trajectories_uncorrected/manifest.csv",
    )
    parser.add_argument("--output-dir", default="output/a_heading_uncertainty_propagated")
    parser.add_argument("--samples-per-trajectory", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=20260710)
    parser.add_argument(
        "--lower-quantile",
        type=float,
        default=0.05,
        help="Lower empirical quantile used for the left/right uncertainty envelope.",
    )
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    args = parser.parse_args()

    if args.samples_per_trajectory <= 0:
        raise ValueError("--samples-per-trajectory must be positive")
    if not 0.0 <= args.lower_quantile < args.upper_quantile <= 1.0:
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

    correction = pd.read_csv(args.correction_summary)
    model, retained = estimate_heading_model(correction, args.lower_quantile, args.upper_quantile)
    source_manifest = pd.read_csv(args.trajectory_manifest, dtype={"dataset": str})
    if source_manifest.empty:
        raise ValueError("The trajectory manifest is empty")

    output_dir = Path(args.output_dir)
    trials_dir = output_dir / "trials"
    figures_dir = output_dir / "figures"
    trials_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    model["samples_per_trajectory"] = args.samples_per_trajectory
    model["random_seed"] = args.random_seed
    model["model_description"] = (
        "Empirical signed heading-drift rates from IQR-filtered straight walks are sampled "
        "and accumulated linearly with each trajectory's PDR path length."
    )

    endpoint_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    plot_rows: list[dict[str, Any]] = []
    signed_rates = retained["heading_change_deg_per_pdr_m"].to_numpy(dtype=float)
    lower_rate = float(model["heading_drift_rate_pdr_m_q_lower"])
    upper_rate = float(model["heading_drift_rate_pdr_m_q_upper"])

    for _, row in source_manifest.iterrows():
        trajectory = pd.read_csv(row["trajectory_csv"])
        if trajectory.empty:
            continue
        relative_path = Path(str(row["trial_id"]).strip("/"))
        output_path = trials_dir / relative_path / "heading_uncertainty_envelope.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        envelope = build_envelope(trajectory, lower_rate, upper_rate)
        envelope.to_csv(output_path, index=False)

        sampled_rates = rng.choice(signed_rates, size=args.samples_per_trajectory, replace=True)
        endpoints = sample_endpoints(trajectory, sampled_rates)
        endpoint_rows.extend(endpoint_records(row, endpoints, sampled_rates))

        nominal = trajectory[["x", "y"]].iloc[-1].to_numpy(dtype=float)
        endpoint_x = endpoints[:, 0]
        endpoint_y = endpoints[:, 1]
        manifest_row = row.to_dict()
        manifest_row.update(
            {
                "heading_uncertainty_envelope_csv": str(output_path),
                "nominal_endpoint_x": float(nominal[0]),
                "nominal_endpoint_y": float(nominal[1]),
                "endpoint_x_q05": float(np.quantile(endpoint_x, 0.05)),
                "endpoint_x_q50": float(np.quantile(endpoint_x, 0.50)),
                "endpoint_x_q95": float(np.quantile(endpoint_x, 0.95)),
                "endpoint_y_q05": float(np.quantile(endpoint_y, 0.05)),
                "endpoint_y_q50": float(np.quantile(endpoint_y, 0.50)),
                "endpoint_y_q95": float(np.quantile(endpoint_y, 0.95)),
                "endpoint_spread_rms_m": float(
                    np.sqrt(np.mean(np.sum((endpoints - nominal) ** 2, axis=1)))
                ),
                "heading_uncertainty_method": "empirical_iqr_filtered_heading_drift_propagation",
            }
        )
        manifest_rows.append(manifest_row)
        plot_rows.append({"seat": row["trajectory_name"], "trajectory": trajectory, "endpoints": endpoints})

    endpoints_df = pd.DataFrame(endpoint_rows)
    propagated_manifest = pd.DataFrame(manifest_rows)
    retained.to_csv(output_dir / "heading_model_retained_straight_trials.csv", index=False)
    pd.DataFrame([model]).to_csv(output_dir / "heading_uncertainty_model.csv", index=False)
    endpoints_df.to_csv(output_dir / "endpoint_samples.csv", index=False)
    propagated_manifest.to_csv(output_dir / "manifest.csv", index=False)
    plot_overview(plot_rows, figures_dir / "a_trajectory_heading_uncertainty_overview.png")
    plot_endpoint_clouds(endpoints_df, figures_dir / "a_endpoint_heading_uncertainty_clouds.png")

    print(f"Straight-walk trials retained after IQR filter: {len(retained)} / {len(correction)}")
    print(f"A trajectories processed: {len(propagated_manifest)}")
    print(f"Endpoint samples created: {len(endpoints_df)}")
    print(
        "Heading model: "
        f"median_abs={model['abs_heading_drift_rate_pdr_m_median']:.6f} deg/PDR-m, "
        f"empirical envelope=[{lower_rate:.6f}, {upper_rate:.6f}] deg/PDR-m"
    )
    print(f"Output directory: {output_dir}")


def estimate_heading_model(
    correction: pd.DataFrame,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    required = {
        "abs_endpoint_x_per_true_m",
        "heading_change_deg",
        "pdr_path_length_m",
    }
    missing = required - set(correction.columns)
    if missing:
        raise ValueError(f"Correction summary is missing columns: {sorted(missing)}")

    valid = correction.dropna(subset=list(required)).copy()
    q1 = float(valid["abs_endpoint_x_per_true_m"].quantile(0.25))
    q3 = float(valid["abs_endpoint_x_per_true_m"].quantile(0.75))
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    retained = valid[valid["abs_endpoint_x_per_true_m"] <= upper_fence].copy()
    retained["heading_change_deg_per_pdr_m"] = (
        retained["heading_change_deg"] / retained["pdr_path_length_m"]
    )
    retained = retained.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["heading_change_deg_per_pdr_m"]
    )
    if retained.empty:
        raise ValueError("No straight-walk trials remained after IQR filtering")

    rates = retained["heading_change_deg_per_pdr_m"]
    model: dict[str, float | int | str] = {
        "calibration_trial_count": int(len(valid)),
        "retained_trial_count": int(len(retained)),
        "outlier_rule": "abs_endpoint_x_per_true_m > Q3 + 1.5 * IQR",
        "lateral_rate_q1_m_per_true_m": q1,
        "lateral_rate_q3_m_per_true_m": q3,
        "lateral_rate_iqr_m_per_true_m": iqr,
        "lateral_rate_iqr_upper_fence_m_per_true_m": upper_fence,
        "heading_drift_rate_pdr_m_mean": float(rates.mean()),
        "heading_drift_rate_pdr_m_std": float(rates.std(ddof=1)),
        "heading_drift_rate_pdr_m_median": float(rates.median()),
        "abs_heading_drift_rate_pdr_m_median": float(rates.abs().median()),
        "heading_drift_rate_pdr_m_q_lower": float(rates.quantile(lower_quantile)),
        "heading_drift_rate_pdr_m_q_upper": float(rates.quantile(upper_quantile)),
        "heading_drift_rate_pdr_m_q25": float(rates.quantile(0.25)),
        "heading_drift_rate_pdr_m_q75": float(rates.quantile(0.75)),
        "heading_drift_rate_pdr_m_q90_abs": float(rates.abs().quantile(0.90)),
    }
    return model, retained


def build_envelope(trajectory: pd.DataFrame, lower_rate: float, upper_rate: float) -> pd.DataFrame:
    nominal = reconstruct_trajectory(trajectory, 0.0)
    lower = reconstruct_trajectory(trajectory, lower_rate)
    upper = reconstruct_trajectory(trajectory, upper_rate)
    return pd.DataFrame(
        {
            "step_index": trajectory["step_index"].to_numpy(),
            "step_length_m": trajectory["step_length_m"].to_numpy(dtype=float),
            "cumulative_pdr_distance_m": nominal["cumulative_pdr_distance_m"],
            "heading_rad_nominal": trajectory["heading_rad"].to_numpy(dtype=float),
            "x_nominal": nominal["x"],
            "y_nominal": nominal["y"],
            "x_lower": lower["x"],
            "y_lower": lower["y"],
            "x_upper": upper["x"],
            "y_upper": upper["y"],
            "heading_drift_rate_lower_deg_per_pdr_m": lower_rate,
            "heading_drift_rate_upper_deg_per_pdr_m": upper_rate,
        }
    )


def reconstruct_trajectory(trajectory: pd.DataFrame, drift_rate_deg_per_pdr_m: float) -> pd.DataFrame:
    headings = trajectory["heading_rad"].to_numpy(dtype=float)
    steps = trajectory["step_length_m"].to_numpy(dtype=float)
    cumulative_distance = np.cumsum(steps)
    heading_error = np.deg2rad(drift_rate_deg_per_pdr_m * cumulative_distance)
    perturbed_headings = headings + heading_error
    x = np.cumsum(steps * np.sin(perturbed_headings))
    y = np.cumsum(steps * np.cos(perturbed_headings))
    return pd.DataFrame(
        {
            "cumulative_pdr_distance_m": cumulative_distance,
            "heading_error_rad": heading_error,
            "x": x,
            "y": y,
        }
    )


def sample_endpoints(trajectory: pd.DataFrame, sampled_rates: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            reconstruct_trajectory(trajectory, float(rate))[["x", "y"]].iloc[-1].to_numpy()
            for rate in sampled_rates
        ],
        dtype=float,
    )


def endpoint_records(row: pd.Series, endpoints: np.ndarray, rates: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_index, (endpoint, rate) in enumerate(zip(endpoints, rates), start=1):
        records.append(
            {
                "dataset": row.get("dataset"),
                "trajectory_name": row.get("trajectory_name"),
                "trial_id": row.get("trial_id"),
                "sample_index": sample_index,
                "sampled_heading_drift_rate_deg_per_pdr_m": float(rate),
                "endpoint_x": float(endpoint[0]),
                "endpoint_y": float(endpoint[1]),
            }
        )
    return records


def plot_overview(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for row in rows:
        color = SEAT_COLORS.get(str(row["seat"]), "tab:gray")
        trajectory = row["trajectory"]
        ax.plot([0.0, *trajectory["x"]], [0.0, *trajectory["y"]], color=color, alpha=0.16, linewidth=0.7)
    ax.scatter([0.0], [0.0], color="black", marker="s", s=35, label="start")
    handles = [plt.Line2D([0], [0], color=color, label=seat) for seat, color in SEAT_COLORS.items()]
    ax.legend(handles=handles + [plt.Line2D([], [], marker="s", color="black", linestyle="", label="start")])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Nominal A-table trajectories with heading-uncertainty model")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_endpoint_clouds(samples: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    for seat, group in samples.groupby("trajectory_name"):
        color = SEAT_COLORS.get(str(seat), "tab:gray")
        ax.scatter(group["endpoint_x"], group["endpoint_y"], color=color, s=3, alpha=0.025)
        ax.scatter(group["endpoint_x"].median(), group["endpoint_y"].median(), color=color, s=36, label=str(seat))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Endpoint uncertainty clouds from propagated heading drift")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.legend(title="seat")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
