from __future__ import annotations

from pathlib import Path

import pandas as pd


ACC_COLUMNS = {
    "Time (s)": "t",
    "X (m/s^2)": "acc_x",
    "Y (m/s^2)": "acc_y",
    "Z (m/s^2)": "acc_z",
}

GYRO_COLUMNS = {
    "Time (s)": "t",
    "X (rad/s)": "gyro_x",
    "Y (rad/s)": "gyro_y",
    "Z (rad/s)": "gyro_z",
}

GRAVITY_COLUMNS = {
    "Time (s)": "t",
    "Gravity X (m/s^2)": "gravity_x",
    "Gravity Y (m/s^2)": "gravity_y",
    "Gravity Z (m/s^2)": "gravity_z",
}

MAG_COLUMNS = {
    "Time (s)": "t",
    "X (µT)": "mag_x",
    "Y (µT)": "mag_y",
    "Z (µT)": "mag_z",
}

def read_accelerometer_csv(path: str | Path) -> pd.DataFrame:
    return _read_phyphox_csv(path, ACC_COLUMNS, "accelerometer")


def read_gyroscope_csv(path: str | Path) -> pd.DataFrame:
    return _read_phyphox_csv(path, GYRO_COLUMNS, "gyroscope")


def read_gravity_csv(path: str | Path) -> pd.DataFrame:
    return _read_phyphox_csv(path, GRAVITY_COLUMNS, "gravity")


def read_magnetometer_csv(path: str | Path) -> pd.DataFrame:
    return _read_phyphox_csv(path, MAG_COLUMNS, "magnetometer")


def find_trial_dirs(raw_data_dir: str | Path) -> list[Path]:
    raw_data_dir = Path(raw_data_dir)
    if not raw_data_dir.exists():
        return []
    return sorted(
        path
        for path in raw_data_dir.rglob("*")
        if path.is_dir()
        and (path / "Accelerometer.csv").exists()
        and (path / "Gyroscope.csv").exists()
    )


def resolve_trial_dir(raw_data_dir: str | Path, trial_id: str) -> Path:
    raw_data_dir = Path(raw_data_dir)
    trial_dir = raw_data_dir / trial_id
    if trial_dir.exists():
        return trial_dir

    matches = [path for path in find_trial_dirs(raw_data_dir) if path.name == trial_id]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        options = ", ".join(str(path.relative_to(raw_data_dir)) for path in matches)
        raise ValueError(f"Multiple trials named {trial_id!r}: {options}")
    raise FileNotFoundError(f"Trial not found: {raw_data_dir / trial_id}")


def output_trial_id(raw_data_dir: str | Path, trial_dir: str | Path) -> str:
    raw_data_dir = Path(raw_data_dir)
    trial_dir = Path(trial_dir)
    try:
        return str(trial_dir.relative_to(raw_data_dir))
    except ValueError:
        return trial_dir.name


def holding_position_from_trial(trial_dir: str | Path) -> str:
    return "pocket" if Path(trial_dir).name.startswith("P") else "hand"


def dataset_name(raw_data_dir: str | Path, trial_dir: str | Path) -> str:
    raw_data_dir = Path(raw_data_dir)
    trial_dir = Path(trial_dir)
    try:
        relative = trial_dir.relative_to(raw_data_dir)
    except ValueError:
        return trial_dir.parent.name
    return relative.parts[0] if len(relative.parts) > 1 else "root"


def _read_phyphox_csv(path: str | Path, column_map: dict[str, str], sensor_name: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{sensor_name} CSV not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    missing = [col for col in column_map if col not in df.columns]
    if missing:
        expected = ", ".join(column_map)
        actual = ", ".join(str(col) for col in df.columns)
        raise ValueError(
            f"Unexpected {sensor_name} CSV columns in {path}. "
            f"Missing: {missing}. Expected: {expected}. Actual: {actual}"
        )
    df = df.rename(columns=column_map)[list(column_map.values())].copy()
    df = df.apply(pd.to_numeric, errors="raise")
    df["t"] = df["t"] - df["t"].iloc[0]
    return df
