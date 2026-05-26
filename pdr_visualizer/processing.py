import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def add_acc_norm(
    acc_df: pd.DataFrame,
    window_size: int,
    step_signal: str = "acc_norm",
    hpf_alpha: float = 0.9,
    gravity_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("acc_smoothing_window must be >= 1")
    if not 0.0 <= hpf_alpha < 1.0:
        raise ValueError("hpf_alpha must satisfy 0.0 <= hpf_alpha < 1.0")
    df = acc_df.copy()
    df["acc_norm"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
    df["acc_norm_smooth"] = df["acc_norm"].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    if step_signal == "acc_norm":
        df["step_signal_raw"] = df["acc_norm"]
    elif step_signal == "vertical_hpf":
        if gravity_df is None:
            raise ValueError("Gravity.csv is required when step_signal is vertical_hpf")
        df["acc_vertical"] = _project_acceleration_to_gravity(df, gravity_df)
        gravity_component = _low_pass(df["acc_vertical"].to_numpy(dtype=float), hpf_alpha)
        df["acc_vertical_gravity"] = gravity_component
        df["acc_vertical_hpf"] = df["acc_vertical"] - gravity_component
        df["step_signal_raw"] = df["acc_vertical_hpf"]
    else:
        raise ValueError("step_signal must be one of: acc_norm, vertical_hpf")

    df["step_signal_smooth"] = df["step_signal_raw"].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()
    return df


def detect_steps(
    acc_df: pd.DataFrame,
    height: float | None,
    distance_s: float,
    prominence: float | None,
    use_custom_algorithm: bool = False,
    window_n: int = 6,
    peak_threshold: float = 0.5,
    pp_threshold: float = 1.0,
) -> pd.DataFrame:
    if use_custom_algorithm:
        return _detect_steps_custom(
            acc_df, window_n, peak_threshold, pp_threshold
        )

    sample_interval = _median_sample_interval(acc_df["t"].to_numpy())
    distance_samples = max(1, int(round(distance_s / sample_interval)))
    peaks, _ = find_peaks(
        acc_df["step_signal_smooth"].to_numpy(),
        height=height,
        distance=distance_samples,
        prominence=prominence,
    )
    return pd.DataFrame(
        {
            "step_index": np.arange(len(peaks), dtype=int),
            "step_time": acc_df["t"].iloc[peaks].to_numpy(),
            "step_sample_index": peaks,
            "step_signal_smooth": acc_df["step_signal_smooth"].iloc[peaks].to_numpy(),
        }
    )

def _detect_steps_custom(
    acc_df: pd.DataFrame,
    N: int,
    peak_threshold: float,
    pp_threshold: float,
) -> pd.DataFrame:
    signal = acc_df["step_signal_smooth"].to_numpy()
    times = acc_df["t"].to_numpy()

    peaks = []
    L = len(signal)
    half_N = N // 2

    for t in range(half_N, L - half_N):
        val = signal[t]

        # Condition 1: Peak exceeds threshold and is local maximum
        if val <= peak_threshold:
            continue

        is_peak = True
        for i in range(-half_N, half_N + 1):
            if i == 0: continue
            if val <= signal[t + i]:
                is_peak = False
                break
        if not is_peak:
            continue

        # Condition 2: Peak-to-peak difference
        diff_prev = val - signal[t - half_N : t]
        diff_next = val - signal[t + 1 : t + half_N + 1]
        if np.max(diff_prev) <= pp_threshold or np.max(diff_next) <= pp_threshold:
            continue

        # Condition 3: Slope (positive before, negative after)
        sum_pos = np.sum(np.diff(signal[t - half_N : t + 1]))
        sum_neg = np.sum(np.diff(signal[t : t + half_N + 1]))

        if (2.0 / N) * sum_pos > 0 and (2.0 / N) * sum_neg < 0:
            peaks.append(t)

    peaks = np.array(peaks, dtype=int)

    # Simple deduplication (keep the highest peak in a small window)
    # Since the custom algorithm might trigger multiple times for the same peak if N is small
    # But usually it's strict enough. We will just return the peaks found.
    return pd.DataFrame(
        {
            "step_index": np.arange(len(peaks), dtype=int),
            "step_time": times[peaks],
            "step_sample_index": peaks,
            "step_signal_smooth": signal[peaks],
        }
    )

def estimate_heading(
    gyro_df: pd.DataFrame,
    gyro_axis: str,
    gyro_sign: float,
    gyro_scale: float,
    initial_heading_rad: float,
    use_bias_correction: bool,
    bias_static_duration_s: float,
    use_smart_pdr: bool = False,
    gravity_df: pd.DataFrame | None = None,
    mag_df: pd.DataFrame | None = None,
    declination_rad: float = 0.0,
    h_cor_t_rad: float = 0.0,
    h_mag_t_rad: float = 0.0,
    mag_correction_gain: float = 0.05,
) -> pd.DataFrame:
    if gyro_sign not in (-1.0, 1.0):
        raise ValueError("gyro_sign must be 1.0 or -1.0")
    if gyro_scale <= 0:
        raise ValueError("gyro_scale must be > 0")
    if use_smart_pdr:
        return _estimate_heading_smart_pdr(
            gyro_df,
            gravity_df,
            mag_df,
            gyro_sign,
            gyro_scale,
            initial_heading_rad,
            use_bias_correction,
            bias_static_duration_s,
            declination_rad,
            h_cor_t_rad,
            h_mag_t_rad,
            mag_correction_gain,
        )

    df = gyro_df.copy()
    axis_col = f"gyro_{gyro_axis}"
    if axis_col not in df.columns:
        raise ValueError("gyro_axis must be one of: x, y, z")

    raw = df[axis_col].to_numpy(dtype=float)
    t = df["t"].to_numpy(dtype=float)
    bias = _initial_bias(raw, t, use_bias_correction, bias_static_duration_s)
    corrected = (raw - bias) * gyro_sign * gyro_scale
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

def _estimate_heading_smart_pdr(
    gyro_df: pd.DataFrame,
    gravity_df: pd.DataFrame | None,
    mag_df: pd.DataFrame | None,
    gyro_sign: float,
    gyro_scale: float,
    initial_heading_rad: float,
    use_bias_correction: bool,
    bias_static_duration_s: float,
    declination_rad: float,
    h_cor_t_rad: float,
    h_mag_t_rad: float,
    mag_correction_gain: float,
) -> pd.DataFrame:
    if gravity_df is None or mag_df is None:
        raise ValueError("Gravity.csv and Magnetometer.csv are required for SmartPDR heading")
    if not 0.0 <= mag_correction_gain <= 1.0:
        raise ValueError("mag_correction_gain must satisfy 0.0 <= gain <= 1.0")

    t = gyro_df["t"].to_numpy(dtype=float)
    gravity = _interpolate_vectors(
        t, gravity_df, ("gravity_x", "gravity_y", "gravity_z")
    )
    gravity_unit = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    gyro = gyro_df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=float)
    vertical_rate = np.sum(gyro * gravity_unit, axis=1)
    bias = _initial_bias(vertical_rate, t, use_bias_correction, bias_static_duration_s)
    corrected = (vertical_rate - bias) * gyro_sign * gyro_scale

    mag = _interpolate_vectors(t, mag_df, ("mag_x", "mag_y", "mag_z"))
    mag_norm = np.linalg.norm(mag, axis=1)
    mag_norm_reference = float(np.median(mag_norm))
    mag_norm_deviation = np.abs(mag_norm - mag_norm_reference)
    pitch = np.arctan2(gravity[:, 1], np.sqrt(gravity[:, 0] ** 2 + gravity[:, 2] ** 2))
    roll = np.arctan2(-gravity[:, 0], gravity[:, 2])
    mag_gcs_x = mag[:, 0] * np.cos(roll) + mag[:, 2] * np.sin(roll)
    mag_gcs_y = (
        -mag[:, 0] * np.sin(pitch) * np.sin(roll)
        - mag[:, 1] * np.cos(pitch)
        + mag[:, 2] * np.sin(pitch) * np.cos(roll)
    )
    mag_absolute = np.unwrap(np.arctan2(-mag_gcs_y, mag_gcs_x) - declination_rad)
    mag_heading = initial_heading_rad + mag_absolute - mag_absolute[0]

    heading_gyro = np.empty_like(corrected)
    heading = np.empty_like(corrected)
    accepted = np.zeros(len(t), dtype=bool)
    mag_heading_error = np.zeros(len(t), dtype=float)
    mag_correction_applied = np.zeros(len(t), dtype=float)
    heading_gyro[0] = initial_heading_rad
    heading[0] = initial_heading_rad
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        predicted = heading[i - 1] + 0.5 * (corrected[i - 1] + corrected[i]) * dt
        heading_gyro[i] = heading_gyro[i - 1] + 0.5 * (corrected[i - 1] + corrected[i]) * dt
        error = _angle_difference(mag_heading[i], predicted)
        mag_delta = _angle_difference(mag_heading[i], mag_heading[i - 1])
        mag_heading_error[i] = error
        correlated = abs(error) <= h_cor_t_rad
        magnetically_stable = abs(mag_delta) <= h_mag_t_rad
        accepted[i] = correlated and magnetically_stable
        mag_correction_applied[i] = mag_correction_gain * error if accepted[i] else 0.0
        heading[i] = predicted + mag_correction_applied[i]

    return pd.DataFrame(
        {
            "t": t,
            "gyro_z": gyro_df["gyro_z"].to_numpy(dtype=float),
            "gyro_z_corrected": gyro_df["gyro_z"].to_numpy(dtype=float),
            "gyro_vertical_raw": vertical_rate,
            "gyro_vertical_corrected": corrected,
            "heading_rad": heading,
            "heading_gyro_rad": heading_gyro,
            "heading_mag_rad": mag_heading,
            "mag_norm_ut": mag_norm,
            "mag_norm_deviation_ut": mag_norm_deviation,
            "mag_heading_error_rad": mag_heading_error,
            "mag_correction_applied_rad": mag_correction_applied,
            "mag_validation_accepted": accepted,
        }
    )

def add_step_lengths(
    steps_df: pd.DataFrame,
    acc_df: pd.DataFrame,
    mode: str,
    fixed_step_length_m: float,
    dynamic_config: dict[str, float],
) -> pd.DataFrame:
    if fixed_step_length_m <= 0:
        raise ValueError("step_length_m must be > 0")
    df = steps_df.copy()
    if mode == "fixed":
        df["heading_time"] = df["step_time"]
        df["step_length_m"] = fixed_step_length_m
        return df
    if mode != "dynamic":
        raise ValueError("step_length_mode must be one of: fixed, dynamic")
    if df.empty:
        df["heading_time"] = []
        df["step_length_m"] = []
        return df

    signal = acc_df["step_signal_smooth"].to_numpy(dtype=float)
    times = acc_df["t"].to_numpy(dtype=float)
    peak_indices = df["step_sample_index"].to_numpy(dtype=int)
    valley_indices: list[int] = []
    for i, peak_index in enumerate(peak_indices):
        start = 0 if i == 0 else peak_indices[i - 1] + 1
        segment = signal[start : peak_index + 1]
        valley_indices.append(start + int(np.argmin(segment)) if len(segment) else peak_index)

    peak_values = signal[peak_indices]
    valley_values = signal[valley_indices]
    peak_to_valley_raw = np.maximum(peak_values - valley_values, 1e-9)
    scale_divisor = float(dynamic_config.get("acc_scale_divisor", 1.0))
    if scale_divisor <= 0:
        raise ValueError("acc_scale_divisor must be > 0")
    peak_to_valley = peak_to_valley_raw / scale_divisor
    threshold = float(dynamic_config["acc_threshold"])
    fourth_root = (
        float(dynamic_config["fourth_root_gamma"]) * np.power(peak_to_valley, 0.25)
        + float(dynamic_config["fourth_root_delta"])
    )
    logarithm = (
        float(dynamic_config["log_gamma"]) * np.log(peak_to_valley)
        + float(dynamic_config["log_delta"])
    )
    step_lengths = np.where(peak_to_valley < threshold, fourth_root, logarithm)
    if bool(dynamic_config.get("calibrate_to_fixed_step_length", False)):
        valid_lengths = step_lengths[np.isfinite(step_lengths) & (step_lengths > 0)]
        if len(valid_lengths):
            step_lengths = step_lengths * (fixed_step_length_m / float(np.median(valid_lengths)))
    step_lengths = np.clip(
        step_lengths,
        float(dynamic_config["min_step_length_m"]),
        float(dynamic_config["max_step_length_m"]),
    )

    df["heading_time"] = times[valley_indices]
    df["acc_peak"] = peak_values
    df["acc_valley"] = valley_values
    df["acc_peak_to_valley"] = peak_to_valley_raw
    df["acc_peak_to_valley_scaled"] = peak_to_valley
    df["step_length_m"] = step_lengths
    return df


def build_trajectory(steps_df: pd.DataFrame, heading_df: pd.DataFrame) -> pd.DataFrame:
    if "step_length_m" not in steps_df.columns:
        raise ValueError("steps_df must include step_length_m")
    step_times = steps_df["step_time"].to_numpy(dtype=float)
    heading_times = steps_df.get("heading_time", steps_df["step_time"]).to_numpy(dtype=float)
    step_lengths = steps_df["step_length_m"].to_numpy(dtype=float)
    headings = np.interp(
        heading_times,
        heading_df["t"].to_numpy(dtype=float),
        heading_df["heading_rad"].to_numpy(dtype=float),
    )
    xs: list[float] = []
    ys: list[float] = []
    x = 0.0
    y = 0.0
    if len(step_lengths) != len(headings):
        raise ValueError(
            f"step_lengths and headings must have the same length, "
            f"got {len(step_lengths)} and {len(headings)}"
        )
    for step_length_m, heading in zip(step_lengths, headings):
        x += step_length_m * np.sin(heading)
        y += step_length_m * np.cos(heading)
        xs.append(x)
        ys.append(y)
    return pd.DataFrame(
        {
            "step_index": steps_df["step_index"].to_numpy(dtype=int),
            "step_time": step_times,
            "heading_time": heading_times,
            "step_length_m": step_lengths,
            "x": xs,
            "y": ys,
            "heading_rad": headings,
        }
    )


def align_heading_to_initial_steps(
    steps_df: pd.DataFrame,
    heading_df: pd.DataFrame,
    step_count: int,
    target_heading_rad: float = 0.0,
) -> pd.DataFrame:
    if step_count < 1:
        raise ValueError("initial_alignment.step_count must be >= 1")
    if steps_df.empty:
        return heading_df.copy()

    aligned = heading_df.copy()
    heading_times = steps_df.get("heading_time", steps_df["step_time"]).to_numpy(dtype=float)
    sample_times = heading_times[: min(step_count, len(heading_times))]
    sampled_headings = np.interp(
        sample_times,
        aligned["t"].to_numpy(dtype=float),
        aligned["heading_rad"].to_numpy(dtype=float),
    )
    offset = _angle_difference(
        float(np.angle(np.mean(np.exp(1j * sampled_headings)))),
        target_heading_rad,
    )
    for column in ("heading_rad", "heading_gyro_rad", "heading_mag_rad"):
        if column in aligned.columns:
            aligned[f"{column}_unaligned"] = aligned[column]
            aligned[column] = aligned[column] - offset
    aligned["heading_alignment_offset_rad"] = offset
    return aligned


def _project_acceleration_to_gravity(acc_df: pd.DataFrame, gravity_df: pd.DataFrame) -> np.ndarray:
    t = acc_df["t"].to_numpy(dtype=float)
    gravity = np.column_stack(
        [
            np.interp(t, gravity_df["t"].to_numpy(dtype=float), gravity_df[col].to_numpy(dtype=float))
            for col in ("gravity_x", "gravity_y", "gravity_z")
        ]
    )
    norm = np.linalg.norm(gravity, axis=1)
    if np.any(norm == 0):
        raise ValueError("Gravity vector contains zero-length samples")
    unit_gravity = gravity / norm[:, None]
    acc = acc_df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float)
    return np.sum(acc * unit_gravity, axis=1)


def _low_pass(values: np.ndarray, alpha: float) -> np.ndarray:
    filtered = np.empty_like(values)
    filtered[0] = values[0]
    for i in range(1, len(values)):
        filtered[i] = alpha * filtered[i - 1] + (1.0 - alpha) * values[i]
    return filtered


def _interpolate_vectors(
    t: np.ndarray, df: pd.DataFrame, columns: tuple[str, str, str]
) -> np.ndarray:
    source_t = df["t"].to_numpy(dtype=float)
    return np.column_stack(
        [np.interp(t, source_t, df[column].to_numpy(dtype=float)) for column in columns]
    )


def _initial_bias(
    angular_rate: np.ndarray,
    t: np.ndarray,
    use_bias_correction: bool,
    duration_s: float,
) -> float:
    if not use_bias_correction:
        return 0.0
    if duration_s <= 0:
        raise ValueError("bias_static_duration_s must be > 0")
    static_mask = t <= t[0] + duration_s
    if not static_mask.any():
        raise ValueError("No samples found in bias_static_duration_s interval")
    return float(angular_rate[static_mask].mean())


def _angle_difference(target: float, reference: float) -> float:
    return float(np.angle(np.exp(1j * (target - reference))))


def _median_sample_interval(t: np.ndarray) -> float:
    if len(t) < 2:
        raise ValueError("At least two samples are required")
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        raise ValueError("Timestamps must be strictly increasing")
    return float(np.median(dt))
