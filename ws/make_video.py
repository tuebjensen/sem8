import argparse
import dataclasses
import json
import math
import os
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature

import original_MRQ.env_preprocessing as env_preprocessing
import original_MRQ.MRQ as MRQ
import original_MRQ.utils as utils
import Sem8Env as _
from eagle2_hg_model.inference_eagle_repo import EagleProcessor
from main import EagleBackbone


class TestRunner:
    def __init__(
        self,
        agent: object,
        env: object,
        eval_eps: int,
        project_name: str,
        output_folder: str = "",
    ):
        self.agent = agent
        self.env = env
        self.eval_eps = eval_eps
        self.project_name = project_name
        self.output_folder = output_folder

    def run(self):
        for i in range(self.eval_eps):
            print(f"Running episode {i + 1}/{self.eval_eps}")
            state, terminated, truncated = self.env.reset(), False, False
            while not (terminated or truncated):
                action = self.agent.select_action(state, use_exploration=False)
                state, _, terminated, truncated = self.env.step(action)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    eval_eps = args.eval_eps
    remove_info = True
    model = EagleBackbone(device, use_last_embedding=args.use_last_embedding)
    env = env_preprocessing.Env(
        args.env,
        args.seed,
        eval_env=False,
        use_maze=args.use_maze,
        use_last_embedding=args.use_last_embedding,
        use_time_based_penalty=args.use_time_based_penalty,
        use_simple_env=args.use_simple_env,
        use_hf_model=args.use_hf_model,
        success_reward=args.success_reward,
        remove_info=remove_info,
        model=model,
        colors=args.colors,
        n_objects=args.n_objects,
        save_video=args.save_video,
        project_name=args.project_name,
        output_folder=args.output_folder,
    )
    agent = MRQ.Agent(
        env.obs_shape,
        env.action_dim,
        env.max_action,
        env.pixel_obs,
        env.discrete,
        device,
        env.history,
        use_last_embedding=args.use_last_embedding,
        use_hf_model=args.use_hf_model,
    )

    if args.use_base_model:
        agent.load_base_model(f"{args.save_folder}/{args.base_project_name}")

    exp = TestRunner(
        agent=agent,
        env=env,
        eval_eps=eval_eps,
        project_name=args.project_name,
        output_folder=args.output_folder,
    )
    exp.run()


def modify_parser(parser):
    # Experiment
    parser.add_argument("--env", default="Gym-HalfCheetah-v4", type=str)
    parser.add_argument("--use_maze", default=False, action="store_true")
    parser.add_argument("--use_last_embedding", default=False, action="store_true")
    parser.add_argument("--use_time_based_penalty", default=False, action="store_true")
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--use_simple_env", default=False, action="store_true")
    parser.add_argument(
        "--use_base_model",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--base_project_name",
        default="base_simple_gr00t",
        type=str,
    )
    parser.add_argument("--success_reward", default=10.0, type=float)
    parser.add_argument(
        "--colors",
        type=str,
        default="yellow",
        help="Comma separated list of colors for simple env. It will put one rectangle of each color on the image.",
    )
    parser.add_argument(
        "--n_objects",
        type=int,
        default=7,
        help="Number of objects to place in the image if not using simple env.",
    )

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--total_timesteps", default=-1, type=int
    )  # Uses default, input to override.
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--save_video", default=False, action="store_true")

    # File name and locations
    parser.add_argument(
        "--project_name", default="", type=str
    )  # Uses default, input to override.
    parser.add_argument("--output_folder", default="./videos", type=str)
    parser.add_argument(
        "--save_folder", default="./checkpoint", type=str
    )  # Uses default, input to override.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_args()
    main(args)
