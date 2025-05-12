# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import defaultdict
import dataclasses
import json
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import tqdm
from safetensors import safe_open
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

import original_MRQ.env_preprocessing as env_preprocessing
import original_MRQ.MRQ as MRQ
import original_MRQ.utils as utils
import Sem8Env as _
from eagle2_hg_model.inference_eagle_repo import EagleProcessor

class MovingAvg:
    def __init__(self):
        self.n = 0
        self.moving_avg = .0

    def add(self, x: float):
        self.n += 1
        self.moving_avg += (x - self.moving_avg) / self.n
    
    def __str__(self) -> str:
        return f"{self.moving_avg} ({self.n})"

    def __repr__(self) -> str:
        return self.__str__()

class EagleBackbone(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.select_layer = 12
        self.config = AutoConfig.from_pretrained(
            "eagle2_hg_model", trust_remote_code=True
        )
        self.device = device
        self.model = AutoModel.from_config(self.config, trust_remote_code=True)
        self.linear = torch.nn.Linear(2048, 1536)
        self.reduce_model()
        self.load_weights()
        self.freeze_and_eval()
        self.model.to(device)
        self.linear.to(device)
        self.pooler = nn.AdaptiveAvgPool1d(25)
        self.processor = EagleProcessor(
            model_path="eagle2_hg_model", max_input_tiles=1, model_spec=None
        )
        self.img_context_token_id = self.processor.get_img_context_token()
        if (
            hasattr(self.model, "vision_model")
            and hasattr(self.model.vision_model, "vision_model")
            and hasattr(self.model.vision_model.vision_model, "vision_towers")
            and len(self.model.vision_model.vision_model.vision_towers) > 1
        ):
            vision_towers = self.model.vision_model.vision_model.vision_towers

            if (
                hasattr(vision_towers[0], "vision_tower")
                and hasattr(vision_towers[0].vision_tower, "vision_model")
                and hasattr(vision_towers[0].vision_tower.vision_model, "encoder")
            ):
                vision_towers[
                    0
                ].vision_tower.vision_model.encoder.gradient_checkpointing = False
                vision_towers[0].vision_tower.vision_model.head = torch.nn.Identity()

            if hasattr(vision_towers[1], "vision_tower"):
                vision_towers[1].vision_tower.head = torch.nn.Identity()

    def load_weights(self):
        backbone_weights = self.get_backbone_weights()
        for key in backbone_weights.keys():
            w = backbone_weights[key]
            if key.startswith("model."):
                k = key.removeprefix("model.")
                if k in self.model.state_dict():
                    self.model.state_dict()[k].copy_(w)
                else:
                    print(f"Key {key} not found in model state dict")
            elif key.startswith("linear."):
                k = key.removeprefix("linear.")
                if k in self.linear.state_dict():
                    self.linear.state_dict()[k].copy_(w)
                else:
                    print(f"Key {key} not found in linear state dict")

    def get_backbone_weights(self):
        # For some reason the module parser thing complains if I don't import this here
        from huggingface_hub import snapshot_download

        path = snapshot_download("nvidia/GR00T-N1-2B", repo_type="model")
        safe_tensors_path = path + "/model.safetensors"
        backbone_tensors: dict[str, torch.Tensor] = {}
        with safe_open(safe_tensors_path, framework="pt", device="cuda") as f:
            keys = f.keys()
            for key in keys:
                # type safety
                assert isinstance(key, str)
                if "backbone." in key:
                    backbone_tensors[key.removeprefix("backbone.")] = f.get_tensor(
                        key
                    )

        return backbone_tensors

    def freeze_and_eval(self):
        self.model.language_model.requires_grad_(False)
        self.model.vision_model.requires_grad_(False)
        self.model.mlp1.requires_grad_(False)
        self.model.language_model.eval()
        self.model.vision_model.eval()
        self.model.mlp1.eval()

    def reduce_model(self):
        self.model.neftune_alpha = None

        # Reduce vision model (Siglip)
        if hasattr(self.model.vision_model, "vision_model") and hasattr(
            self.model.vision_model.vision_model, "head"
        ):
            self.model.vision_model.vision_model.head = torch.nn.Identity()

        # Remove language modelling head and remove layers
        self.model.language_model.lm_head = torch.nn.Identity()
        while len(self.model.language_model.model.layers) > self.select_layer:
            self.model.language_model.model.layers.pop(-1)

    def get_embeddings(
        self,
        reproject_vision: bool,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        output_hidden_states=None,
        skip_llm=False,
        img_context_token_id=None,
    ) -> torch.Tensor:
        assert pixel_values is not None
        assert img_context_token_id is not None

        vit_embeds = self.model.extract_feature(pixel_values)

        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == img_context_token_id
        assert selected.sum() != 0

        embeds_to_scatter = vit_embeds.reshape(-1, C).to(
            input_embeds.device, input_embeds.dtype
        )
        input_embeds[selected] = embeds_to_scatter
        input_embeds = input_embeds.reshape(B, N, C)

        # return hidden_states
        embeddings = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeddings = embeddings.hidden_states[-1]
        pooled_embeddings = self.pooler(
            embeddings[:, :-25, :].transpose(1, 2)
        ).transpose(1, 2)
        return torch.cat([pooled_embeddings, embeddings[:, :25, :]], dim=1)

    def prepare_message(self, message: list):
        # Message could look like [{"role": "system", "content": "SYSTEM MESSAGE HERE" }, {"role": "user", "image": [{"np_array": np.ndarray(image_data)}], "content": "USER MESSAGE HERE" }]
        inputs = self.processor.prepare_input({"prompt": message})
        return BatchFeature(inputs).to(self.device)

    def forward(self, inputs: BatchFeature):
        embeddings = self.get_embeddings(
            reproject_vision=False,
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            img_context_token_id=self.img_context_token_id,
        )
        return self.linear(embeddings)


@dataclasses.dataclass
class DefaultExperimentArguments:
    Atari_total_timesteps: int = 25e5
    Atari_eval_freq: int = 1e5

    Dmc_total_timesteps: int = 5e5
    Dmc_eval_freq: int = 5e3

    Gym_total_timesteps: int = 1e6  # 1e6
    Gym_eval_freq: int = 5e3

    Frozen_total_timesteps: int = 1e6  # 1e6
    Frozen_eval_freq: int = 5e3

    Sem8_total_timesteps: int = 1e6  # 1e6
    Sem8_eval_freq: int = 5e4  # 5e3

    def __post_init__(self):
        utils.enforce_dataclass_type(self)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    default_arguments = DefaultExperimentArguments()
    env_type = args.env.split("-", 1)[0]
    if args.total_timesteps == -1:
        args.total_timesteps = default_arguments.__dict__[f"{env_type}_total_timesteps"]
    if args.eval_freq == -1:
        args.eval_freq = default_arguments.__dict__[f"{env_type}_eval_freq"]

    # File name and make folders
    if args.project_name == "":
        args.project_name = f"MRQ+{args.env}+{args.seed}"
    if not os.path.exists(args.eval_folder):
        os.makedirs(args.eval_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if args.save_experiment and not os.path.exists(
        f"{args.save_folder}/{args.project_name}"
    ):
        os.makedirs(f"{args.save_folder}/{args.project_name}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.load_experiment:
        exp = load_experiment(args.save_folder, args.project_name, device, args)
    else:
        eval_eps = args.eval_eps
        eval_data_folder_path = ""
        remove_info = True
        model = None  # EagleBackbone(device)
        if args.eval_data_folder != "":
            eval_data_folder_path = os.path.abspath(args.eval_data_folder)
            # Make sure that data directory exists
            assert os.path.exists(eval_data_folder_path), "eval_data_dir does not exist"
            with open(os.path.join(eval_data_folder_path, "meta_data.json"), "r") as f:
                meta_data = json.load(f)
            eval_eps = min(len(meta_data), eval_eps)
            model = EagleBackbone(device)

        env = env_preprocessing.Env(
            args.env,
            args.seed,
            eval_env=False,
            eval_data_dir=eval_data_folder_path,
            remove_info=remove_info,
            model=model,
        )
        eval_env = env_preprocessing.Env(
            args.env,
            args.seed + 100,
            eval_eps=eval_eps,
            eval_env=True,
            eval_data_dir=eval_data_folder_path,
            remove_info=remove_info,
            model=model,
        )  # +100 to make sure the seed is different.

        agent = MRQ.Agent(
            env.obs_shape,
            env.action_dim,
            env.max_action,
            env.pixel_obs,
            env.discrete,
            device,
            env.history,
        )

        logger = utils.Logger(f"{args.log_folder}/{args.project_name}.txt")

        exp = OnlineExperiment(
            agent,
            env,
            eval_env,
            logger,
            [],
            0,
            args.total_timesteps,
            0,
            args.eval_freq,
            eval_eps,
            args.eval_folder,
            args.project_name,
            args.save_experiment,
            args.save_freq,
            args.save_folder,
        )

    exp.logger.title("Experiment")
    exp.logger.log_print(f"Algorithm:\t{exp.agent.name}")
    exp.logger.log_print(f"Env:\t\t{exp.env.env_name}")
    exp.logger.log_print(f"Seed:\t\t{exp.env.seed}")

    exp.logger.title("Environment hyperparameters")
    if hasattr(exp.env.env, "hp"):
        exp.logger.log_print(exp.env.env.hp)
    exp.logger.log_print(f"Obs shape:\t\t{exp.env.obs_shape}")
    exp.logger.log_print(f"Action dim:\t\t{exp.env.action_dim}")
    exp.logger.log_print(f"Discrete actions:\t{exp.env.discrete}")
    exp.logger.log_print(f"Pixel observations:\t{exp.env.pixel_obs}")

    exp.logger.title("Agent hyperparameters")
    exp.logger.log_print(exp.agent.hp)
    exp.logger.log_print("-" * 40)

    exp.run()


class OnlineExperiment:
    def __init__(
        self,
        agent: object,
        env: object,
        eval_env: object,
        logger: object,
        evals: list,
        t: int,
        total_timesteps: int,
        time_passed: float,
        eval_freq: int,
        eval_eps: int,
        eval_folder: str,
        project_name: str,
        save_full: bool = False,
        save_freq: int = 1e3,
        save_folder: str = "",
        timings: defaultdict[str, MovingAvg] | None=None,
    ):
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.evals = evals

        self.logger = logger
        self.t = t
        self.time_passed = time_passed
        self.start_time = time.time()

        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.eval_eps = eval_eps

        self.eval_folder = eval_folder
        self.project_name = project_name
        self.save_full = save_full
        self.save_freq = save_freq
        self.save_folder = save_folder

        self.init_timestep = True

        self.timings = defaultdict(MovingAvg) if timings is None else timings

    def run(self):
        t = time.time()
        state = self.env.reset()
        self.timings["reset"].add(time.time() - t)

        while self.t <= self.total_timesteps:
            self.maybe_evaluate()

            if (
                self.save_full
                and self.t % self.save_freq == 0
                and not self.init_timestep
            ):
                t = time.time()
                save_experiment(self)
                self.timings["save"].add(time.time() - t)

            t = time.time()
            action = self.agent.select_action(state)
            if action is None:
                action = self.env.action_space.sample()
            else:
                self.timings["select_action"].add(time.time() - t)

            t = time.time()
            next_state, reward, terminated, truncated = self.env.step(action)
            self.agent.replay_buffer.add(
                state, action, next_state, reward, terminated, truncated
            )
            state = next_state
            self.timings["step"].add(time.time() - t)

            t = time.time()
            self.agent.train()
            self.timings["train"].add(time.time() - t)

            if terminated or truncated:
                self.logger.log_print(
                    f"Total T: {self.t + 1}, "
                    f"Episode Num: {self.env.ep_num}, "
                    f"Episode T: {self.env.ep_timesteps}, "
                    f"Reward: {self.env.ep_total_reward:.3f}, "
                    f"Timings: {dict(self.timings)}"
                )
                t = time.time()
                state = self.env.reset()
                self.timings["reset"].add(time.time() - t)

            self.t += 1
            self.init_timestep = False

    def maybe_evaluate(self):
        if self.t % self.eval_freq != 0:
            return

        # We save after evaluating, this avoids re-evaluating immediately after loading an experiment.
        if self.t != 0 and self.init_timestep:
            return
        total_reward = np.zeros(self.eval_eps)
        for ep in tqdm.tqdm(range(self.eval_eps)):
            print("Evaluating episode", ep + 1)

            t = time.time()
            state, terminated, truncated = self.eval_env.reset(), False, False
            self.timings["eval_reset"].add(time.time() - t)

            while not (terminated or truncated):
                t = time.time()
                action = self.agent.select_action(state, use_exploration=False)
                self.timings["eval_select_action"].add(time.time() - t)

                t = time.time()
                state, _, terminated, truncated = self.eval_env.step(action)
                self.timings["eval_step"].add(time.time() - t)

            total_reward[ep] = self.eval_env.ep_total_reward

        self.evals.append(total_reward.mean())

        self.logger.title(
            f"Evaluation at {self.t} time steps\n"
            f"Average total reward over {self.eval_eps} episodes: {total_reward.mean():.3f}\n"
            f"Total time passed: {round((time.time() - self.start_time + self.time_passed) / 60.0, 2)} minutes\n"
            f"Timings: {dict(self.timings)}",
        )

        t = time.time()
        np.savetxt(
            f"{self.eval_folder}/{self.project_name}.txt", self.evals, fmt="%.14f"
        )
        self.timings["eval_save"].add(time.time() - t)


def save_experiment(exp: OnlineExperiment):
    # Save experiment settings
    exp.time_passed += time.time() - exp.start_time
    var_dict = {k: exp.__dict__[k] for k in ["t", "eval_freq", "eval_eps"]}
    var_dict["time_passed"] = exp.time_passed + time.time() - exp.start_time
    var_dict["np_seed"] = np.random.get_state()
    var_dict["torch_seed"] = torch.get_rng_state()
    var_dict["timings"] = exp.timings
    np.save(f"{exp.save_folder}/{exp.project_name}/exp_var.npy", var_dict)
    # Save eval
    np.savetxt(f"{exp.save_folder}/{exp.project_name}.txt", exp.evals, fmt="%.14f")
    # Save envs
    pickle.dump(
        exp.env, file=open(f"{exp.save_folder}/{exp.project_name}/env.pickle", "wb")
    )
    pickle.dump(
        exp.eval_env,
        file=open(f"{exp.save_folder}/{exp.project_name}/eval_env.pickle", "wb"),
    )
    # Save agent
    exp.agent.save(f"{exp.save_folder}/{exp.project_name}")

    exp.logger.title("Saved experiment")


def load_experiment(
    save_folder: str, project_name: str, device: torch.device, args: object
):
    # Load experiment settings
    exp_dict = np.load(
        f"{save_folder}/{project_name}/exp_var.npy", allow_pickle=True
    ).item()
    # This is not sufficient to guarantee the experiment will run exactly the same,
    # however, it does mean the original seed is not reused.
    np.random.set_state(exp_dict["np_seed"])
    torch.set_rng_state(exp_dict["torch_seed"])
    # Load eval
    evals = np.loadtxt(f"{save_folder}/{project_name}.txt").tolist()
    # Load envs
    env = pickle.load(open(f"{save_folder}/{project_name}/env.pickle", "rb"))
    eval_env = pickle.load(open(f"{save_folder}/{project_name}/eval_env.pickle", "rb"))
    # Load agent
    agent_dict = np.load(
        f"{save_folder}/{project_name}/agent_var.npy", allow_pickle=True
    ).item()
    agent = MRQ.Agent(
        env.obs_shape,
        env.action_dim,
        env.max_action,
        env.pixel_obs,
        env.discrete,
        device,
        env.history,
        dataclasses.asdict(agent_dict["hp"]),
    )
    agent.load(f"{save_folder}/{project_name}")

    logger = utils.Logger(f"{args.log_folder}/{args.project_name}.txt")
    logger.title(f"Loaded experiment\nStarting from: {exp_dict['t']} time steps.")

    return OnlineExperiment(
        agent,
        env,
        eval_env,
        logger,
        evals,
        exp_dict["t"],
        args.total_timesteps,
        exp_dict["time_passed"],
        exp_dict["eval_freq"],
        exp_dict["eval_eps"],
        args.eval_folder,
        args.project_name,
        args.save_experiment,
        args.save_freq,
        args.save_folder,
        timings=exp_dict["timings"],
    )


def modify_parser(parser):
    # Experiment
    parser.add_argument("--env", default="Gym-HalfCheetah-v4", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--total_timesteps", default=-1, type=int
    )  # Uses default, input to override.
    parser.add_argument("--device", default="cuda", type=str)
    # Evaluation
    parser.add_argument(
        "--eval_freq", default=-1, type=int
    )  # Uses default, input to override.
    parser.add_argument("--eval_eps", default=10, type=int)
    parser.add_argument("--eval_data_folder", default="", type=str)

    # File name and locations
    parser.add_argument(
        "--project_name", default="", type=str
    )  # Uses default, input to override.
    parser.add_argument("--eval_folder", default="./evals", type=str)
    parser.add_argument("--log_folder", default="./logs", type=str)
    parser.add_argument("--save_folder", default="./checkpoint", type=str)
    # Experiment checkpointing
    parser.add_argument(
        "--save_experiment",
        default=False,
        action=argparse.BooleanOptionalAction,
        type=bool,
    )
    parser.add_argument("--save_freq", default=1e5, type=int)
    parser.add_argument(
        "--load_experiment",
        default=False,
        action=argparse.BooleanOptionalAction,
        type=bool,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modify_parser(parser)
    args = parser.parse_args()
    main(args)
