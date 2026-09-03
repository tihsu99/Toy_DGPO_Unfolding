from __future__ import annotations

import argparse

from .pipeline import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the toy DGPO forward-folding closure pipeline")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("run", "train", "evaluate", "diagnose", "closure"):
        command = subparsers.add_parser(mode)
        command.add_argument("--config", default="config/default.yaml", help="YAML experiment configuration")
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the configured device")
        command.add_argument("--output-dir", help="Override the configured output directory")
    for mode in ("spin-run", "spin-train", "spin-evaluate", "spin-passive"):
        command = subparsers.add_parser(mode)
        command.add_argument("--spin-config", default="config/spin_matrix.yaml", help="Spin-matrix study configuration")
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the configured device")
        command.add_argument("--output-dir", help="Override the spin-matrix output directory")
        command.add_argument("--cnn-output-dir", help="Override the frozen C_nn study directory")
    for mode in ("spin-conditional-run", "spin-conditional-train", "spin-conditional-evaluate"):
        command = subparsers.add_parser(mode)
        command.add_argument(
            "--spin-config", default="config/spin_conditional.yaml",
            help="Conditional-spin study configuration",
        )
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override the configured device")
        command.add_argument("--output-dir", help="Override the conditional-spin output directory")
        command.add_argument("--cnn-output-dir", help="Override the frozen C_nn study directory")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.mode.startswith("spin-conditional-"):
        from .spin_conditional import run_spin_conditional
        run_spin_conditional(
            arguments.spin_config, arguments.mode, arguments.device,
            arguments.output_dir, arguments.cnn_output_dir,
        )
    elif arguments.mode.startswith("spin-"):
        from .spin_matrix import run_spin_matrix
        run_spin_matrix(
            arguments.spin_config, arguments.mode, arguments.device,
            arguments.output_dir, arguments.cnn_output_dir,
        )
    else:
        run(arguments.config, arguments.mode, arguments.device, arguments.output_dir)


if __name__ == "__main__":
    main()
