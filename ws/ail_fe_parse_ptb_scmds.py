import argparse
import json

from ail_fe_main_scmds import SCmd


def modify_parser(parser: argparse._ArgumentGroup):
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the experiments without extension",
    )


def get_scmd(experiment, folder_name):
    ts2vec_path = experiment["encoder_path"]
    return SCmd(
        program="srun",
        opts=["-J", "parsing", f"--gres=gpu:1", "--mem-per-gpu=30G"],
        python_module="parse_ptb",
        python_args=[
            "--ts2vec_path",
            ts2vec_path,
            "--out_folder",
            f"./data/ptb-xl/{folder_name}/",
        ],
    )


def get_scmds(args: argparse.Namespace):
    with open(f"{args.experiment_path}.json", "r") as f:
        experiments_dict = json.load(f)
        experiments = experiments_dict["experiments"]
        folder_name = experiments_dict["folder_name"]

    return [get_scmd(experiment, folder_name) for experiment in experiments]
