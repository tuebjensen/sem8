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
    hd = experiment["tsencoder_hidden_dim"]
    d = experiment["tsencoder_depth"]
    ed = experiment["ts_embedding_dim"]
    epochs = experiment["epochs"]
    name = f"ts2vec_hd{hd}_d{d}_ed{ed}_ep{epochs}"
    return SCmd(
        opts=["-J", name, f"--gres=gpu:1", "--mem-per-gpu=30G"],
        python_module="train_ts2vec",
        python_args=[
            "--model_name",
            name,
            "--tsencoder_depth",
            d,
            "--tsencoder_hidden_dim",
            hd,
            "--ts_embedding_dim",
            ed,
            "--epochs",
            epochs,
            "--folder_name",
            folder_name,
        ],
    )


def get_scmds(args: argparse.Namespace):
    with open(f"{args.experiment_path}.json", "r") as f:
        experiments_dict = json.load(f)
        folder_name = experiments_dict["folder_name"]
        experiments = experiments_dict["experiments"]

    return [get_scmd(experiment, folder_name) for experiment in experiments]
