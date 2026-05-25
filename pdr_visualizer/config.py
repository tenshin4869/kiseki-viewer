from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "preprocessing": {"acc_smoothing_window": 10},
    "step_detection": {
        "height": 12.0,
        "distance_s": 0.35,
        "prominence": 0.4,
        "height_by_dataset": {},
    },
    "heading": {
        "gyro_axis": "z",
        "gyro_sign": -1.0,
        "initial_heading_rad": 0.0,
        "use_bias_correction": True,
        "bias_static_duration_s": 2.0,
    },
    "pdr": {
        "step_length_m": 0.65,
        "step_length_by_dataset": {},
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
