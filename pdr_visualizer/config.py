from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "preprocessing": {
        "acc_smoothing_window": 10,
        "step_signal": "acc_norm",
        "hpf_alpha": 0.9,
    },
    "step_detection": {
        "height": 12.0,
        "distance_s": 0.35,
        "prominence": 0.4,
        "height_by_dataset": {},
    },
    "heading": {
        "gyro_axis": "z",
        "gyro_sign": -1.0,
        "gyro_scale": 1.0,
        "initial_heading_rad": 0.0,
        "use_bias_correction": True,
        "bias_static_duration_s": 1.0,
        "use_smart_pdr": False,
        "declination_deg": 0.0,
        "h_cor_t_deg": 5.0,
        "h_mag_t_deg": 2.0,
        "mag_correction_gain": 0.05,
    },
    "pdr": {
        "step_length_mode": "fixed",
        "step_length_m": 0.65,
        "step_length_by_dataset": {},
        "dynamic_step_length": {
            "acc_scale_divisor": 1.0,
            "acc_threshold": 3.23,
            "fourth_root_gamma": 1.479,
            "fourth_root_delta": -1.259,
            "log_gamma": 1.131,
            "log_delta": 0.159,
            "calibrate_to_fixed_step_length": False,
            "min_step_length_m": 0.25,
            "max_step_length_m": 1.2,
        },
    },
    "comparison": {
        "simple": {
            "acc_smoothing_window": 10,
            "height": 12.0,
            "height_by_dataset": {},
            "distance_s": 0.35,
            "prominence": 0.4,
            "gyro_axis": "z",
            "gyro_sign": -1.0,
            "gyro_scale": 1.0,
            "initial_heading_rad": 0.0,
            "use_bias_correction": True,
            "bias_static_duration_s": 2.0,
            "step_length_m": 0.65,
            "step_length_by_dataset": {},
        },
    },
    "visualization": {"figure_dpi": 200, "equal_axis": True, "show_grid": True},
    "paths": {"raw_data_dir": "data", "output_dir": "output"},
}


def load_config(path: str | Path) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    path = Path(path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) if yaml is not None else _load_simple_yaml(f.read())
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a YAML mapping: {path}")
        _deep_update(config, loaded)
    return config


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line:
            raise ValueError("Unsupported config.yaml format without PyYAML installed")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value == "":
            parent[key] = {}
            stack.append((indent, parent[key]))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    lower = value.lower()
    if lower in {"null", "none"}:
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return float(value) if "." in value else int(value)
    except ValueError:
        return value
