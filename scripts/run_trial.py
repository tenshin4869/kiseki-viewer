from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pdr_visualizer.config import load_config
from pdr_visualizer.trial import run_trial


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple Data2 PDR for one trial.")
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = run_trial(args.trial_id, config)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
