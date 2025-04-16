import argparse
import json
import os

from ail_fe_main_scmds import SCmd


def modify_parser(parser: argparse._ArgumentGroup):
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the experiments without extension",
    )


def get_scmd(experiment, folder_name):
    el = experiment["ts_embedding_length"]
    nl = experiment["mapper_num_layers"]
    pl = experiment["prefix_length"]
    ep = experiment["epochs"]
    encoder_path = experiment["encoder_path"]
    # "./data/ts2vec_dim_collapse/ts2vec_hd64_d5_ed512_ep2_snapshot.pt"
    encoder_name = (
        os.path.basename(encoder_path).split(".")[0].removesuffix("_snapshot")
    )
    encoder_params = encoder_name.split("_")
    encoder_hd = encoder_params[1]
    encoder_d = encoder_params[2]
    encoder_ed = encoder_params[3]
    encoder_epochs = encoder_params[4]
    model_index = experiment["index"]

    model_name = f"tscap_el{el}_nl{nl}_pl{pl}_ep{ep}_{encoder_hd}_{encoder_d}_{encoder_ed}_e{encoder_epochs}_{model_index}"
    # model_name = f"tscap_el{el}_nl{nl}_pl{pl}_{encoder_hd}_{encoder_d}_{encoder_ed}_e{encoder_epochs}"
    test_path = f"./data/ptb-xl/{folder_name}/{encoder_name}_parsed_ptb_test.pkl"
    tscap_path = f"./data/{folder_name}/{model_name}_snapshot.pt"
    out_dir = f"./metrics/{folder_name}"
    return SCmd(
        program="srun",
        opts=["-J", model_name, f"--gres=gpu:1", "--mem-per-gpu=60G"],
        python_module="predict",
        python_args=[
            "--test_path",
            test_path,
            "--snapshot_path",
            tscap_path,
            "--out_dir",
            out_dir,
            "--out_name",
            f"{model_name}",
            "--ts_embedding_length",
            el,
            "--mapper_num_layers",
            nl,
            "--prefix_length",
            pl,
            "--n_iter",
            20,
            "--metrics",
        ],
    )


def get_scmds(args: argparse.Namespace):
    with open(f"{args.experiment_path}.json", "r") as f:
        experiments_dict = json.load(f)
        experiments = experiments_dict["experiments"]
        folder_name = experiments_dict["folder_name"]
        for i, experiment in enumerate(experiments):
            experiment["index"] = i

    return [get_scmd(experiment, folder_name) for experiment in experiments]
