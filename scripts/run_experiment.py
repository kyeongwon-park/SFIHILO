from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sfihilo.experiment import load_config, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SFIHILO supplementary experiments.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "smoke.json"),
        help="Path to a JSON experiment config.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    config.setdefault("output_dir", str(Path("results") / config_path.stem))
    paths = run_experiment(config)
    print("Saved:")
    for key, value in paths.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
