import argparse
from typing import Literal


class SCmd:
    def __init__(
        self,
        opts: list[str],
        python_module: str | None = None,
        python_args: list[str | int] = [],
    ):
        self.opts = opts
        self.python_module = python_module
        self.python_args = [str(python_args) for python_args in python_args]


def modify_parser(parser: argparse._ArgumentGroup):
    parser.add_argument(
        "--gpu",
        "-g",
        type=int,
        required=True,
        help="The number of GPUs to request in Slurm.",
    )


def get_scmds(args: argparse.Namespace) -> list[SCmd]:
    return [
        SCmd(
            opts=([f"--gres=gpu:{args.gpu}", "--mem-per-gpu=50G"]),
        )
    ]


# run.sh --keep-jobs --scmds-from=main_run_scmds.py -g -n
