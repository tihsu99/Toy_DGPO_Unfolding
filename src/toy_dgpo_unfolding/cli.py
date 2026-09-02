from __future__ import annotations

import argparse

from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the toy DGPO Bayesian-unfolding closure pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "train", "evaluate"):
        command = subparsers.add_parser(mode)
        command.add_argument("--config", default="config/default.yaml", help="YAML experiment configuration")
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the configured device")
        command.add_argument("--output-dir", help="Override the configured output directory")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    run(arguments.config, arguments.mode, arguments.device, arguments.output_dir)


if __name__ == "__main__":
    main()

