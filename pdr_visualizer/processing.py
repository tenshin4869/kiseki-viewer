from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def add_acc_norm(acc_df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("acc_smoothing_window must be >= 1")
    df = acc_df.copy()
    df["acc_norm"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["acc_norm_smooth"] = df["acc_norm"].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    return df


def detect_steps(
    acc_df: pd.DataFrame,
    height: float | None,
    distance_s: float,
    prominence: float | None,
) -> pd.DataFrame:
    sample_interval = _median_sample_interval(acc_df["t"].to_numpy())
    distance_samples = max(1, int(round(distance_s / sample_interval)))
    peaks, _ = find_peaks(
        acc_df["acc_norm_smooth"].to_numpy(),
        height=height,
        distance=distance_samples,
        prominence=prominence,
    )
    return pd.DataFrame(
        {
            "step_index": np.arange(len(peaks), dtype=int),
            "step_time": acc_df["t"].iloc[peaks].to_numpy(),
            "acc_norm_smooth": acc_df["acc_norm_smooth"].iloc[peaks].to_numpy(),
        }
    )


def estimate_heading(
    gyro_df: pd.DataFrame,
    gyro_axis: str,
    gyro_sign: float,
    initial_heading_rad: float,
    use_bias_correction: bool,
    bias_static_duration_s: float,
) -> pd.DataFrame:
    df = gyro_df.copy()
    axis_col = f"gyro_{gyro_axis}"
    if axis_col not in df.columns:
        raise ValueError("gyro_axis must be one of: x, y, z")
    if gyro_sign not in (-1.0, 1.0):
        raise ValueError("gyro_sign must be 1.0 or -1.0")

    raw = df[axis_col].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    if use_bias_correction:
        static_mask = t <= bias_static_duration_s
        if not static_mask.any():
            raise ValueError("No samples found in bias_static_duration_s interval")
        bias = float(raw[static_mask].mean())
    else:
        bias = 0.0

    corrected = (raw - bias) * gyro_sign
    heading = np.empty_like(corrected)
    heading[0] = initial_heading_rad
    if len(heading) > 1:
        heading[1:] = initial_heading_rad + np.cumsum(
            0.5 * (corrected[:-1] + corrected[1:]) * np.diff(t)
        )
    return pd.DataFrame(
        {
            "t": t,
            axis_col: raw,
            f"{axis_col}_corrected": corrected,
            "heading_rad": heading,
        }
    )


def build_trajectory(steps_df: pd.DataFrame, heading_df: pd.DataFrame, step_length_m: float) -> pd.DataFrame:
    if step_length_m <= 0:
        raise ValueError("step_length_m must be > 0")
    step_times = steps_df["step_time"].to_numpy(dtype=float)
    headings = np.interp(
        step_times,
        heading_df["t"].to_numpy(dtype=float),
        heading_df["heading_rad"].to_numpy(dtype=float),
    )
    xs: list[float] = []
    ys: list[float] = []
    x = 0.0
    y = 0.0
    for heading in headings:
        x += step_length_m * np.sin(heading)
        y += step_length_m * np.cos(heading)
        xs.append(x)
        ys.append(y)
    return pd.DataFrame(
        {
            "step_index": steps_df["step_index"].to_numpy(dtype=int),
            "step_time": step_times,
            "x": xs,
            "y": ys,
            "heading_rad": headings,
        }
    )


def _median_sample_interval(t: np.ndarray) -> float:
    if len(t) < 2:
        raise ValueError("At least two samples are required")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Timestamps must be strictly increasing")
    return float(np.median(dt))
