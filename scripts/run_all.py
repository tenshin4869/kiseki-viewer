from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.io import find_trial_dirs
from pdr_visualizer.overlay import plot_all_trials
from pdr_visualizer.trial import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Data2 trials and plot overlay.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    trials = find_trial_dirs(config["paths"]["raw_data_dir"])
    for trial_path in trials:
        trial_id = str(trial_path.relative_to(config["paths"]["raw_data_dir"]))
        outputs = run_trial(trial_id, config)
        print(f"trial: {trial_path.name} -> {outputs['trajectory']}")
    overlays = plot_all_trials(config)
    for name, path in overlays.items():
        print(f"overlay {name}: {path}")


if __name__ == "__main__":
    main()
