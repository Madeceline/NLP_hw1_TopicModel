"""Run from project root: python -m scripts.run_experiments --config configs/default.yaml."""
from __future__ import annotations
import argparse
from src.evaluation.experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=".")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_experiments(args.config, args.output, resume=not args.no_resume)


if __name__ == "__main__":
    main()
