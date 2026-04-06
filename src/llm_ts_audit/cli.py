from __future__ import annotations

import argparse

from llm_ts_audit.config import load_config
from llm_ts_audit.evaluation.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM time-series adversarial audit runner.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()
    config = load_config(args.config)
    output_dir = run_experiment(config)
    print(f"Experiment completed. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()

