from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.overlay import plot_all_trials


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Data2 overlay trajectories.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = plot_all_trials(config)
    for name, path in outputs.items():
        print(f"overlay {name}: {path}")


if __name__ == "__main__":
    main()
