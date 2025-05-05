import copy
import dataclasses
from typing import Callable, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import MRQ.buffer as buffer
from MRQ.models import Encoder, Policy, Value


def set_instance_vars(hp: dataclasses.dataclass, c: object):
    for field in dataclasses.fields(hp):
        c.__dict__[field.name] = getattr(hp, field.name)


@dataclasses.dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    buffer_size: int = int(1e4)  # 1e6
    discount: float = 0.99
    target_update_freq: int = 250

    # Exploration
    buffer_size_before_training: int = int(1e2)  # 10e3
    exploration_noise: float = (
        0.2 * 0.5
    )  # 0.5 since we are using discrete actions which range from 0 to 1 instead of -1 to 1

    # TD3
    target_policy_noise: float = 0.2 * 0.5  # 0.2
    noise_clip: float = 0.3 * 0.5  # 0.3

    # Encoder Loss
    dyn_weight: float = 1
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Replay Buffer (LAP)
    prioritized: bool = True
    alpha: float = 0.4
    min_priority: float = 1
    enc_horizon: int = 5
    Q_horizon: int = 3

    # Encoder Model
    zs_dim: int = 512
    zsa_dim: int = 512
    za_dim: int = 256
    enc_hdim: int = 512
    enc_activation: Callable = F.elu
    enc_lr: float = 1e-4
    enc_wd: float = 1e-4
    pixel_augs: bool = True

    # Value Model
    value_hdim: int = 512
    value_activation: Callable = F.elu
    value_lr: float = 3e-4
    value_wd: float = 1e-4
    value_grad_clip: float = 20

    # Policy Model
    policy_hdim: int = 512
    policy_activation: Callable = F.relu
    policy_lr: float = 3e-4
    policy_wd: float = 1e-4
    gumbel_tau: float = 10
    policy_action_weight: float = 1e-5

    # Reward model
    num_bins: int = 65
    lower: float = -10
    upper: float = 10


class TwoHot:
    def __init__(
        self,
        device: torch.device,
        lower: float = -10,
        upper: float = 10,
        num_bins: int = 101,
    ):
        self.bins = torch.linspace(lower, upper, num_bins, device=device)
        self.bins = self.bins.sign() * (self.bins.abs().exp() - 1)  # symexp
        self.num_bins = num_bins

    def transform(self, x: torch.Tensor):
        diff = x - self.bins.reshape(1, -1)
        diff = diff - 1e8 * (torch.sign(diff) - 1)
        ind = torch.argmin(diff, 1, keepdim=True)

        lower = self.bins[ind]
        upper = self.bins[(ind + 1).clamp(0, self.num_bins - 1)]
        weight = (x - lower) / (upper - lower)

        two_hot = torch.zeros(x.shape[0], self.num_bins, device=x.device)
        two_hot.scatter_(1, ind, 1 - weight)
        two_hot.scatter_(1, (ind + 1).clamp(0, self.num_bins), weight)
        return two_hot

    def inverse(self, x: torch.Tensor):
        return (F.softmax(x, dim=-1) * self.bins).sum(-1, keepdim=True)

    def cross_entropy_loss(self, pred: torch.Tensor, target: torch.Tensor):
        pred = F.log_softmax(pred, dim=-1)
        target = self.transform(target)
        return -(target * pred).sum(-1, keepdim=True)


def masked_mse(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    return (F.mse_loss(x, y, reduction="none") * mask).mean()


def multi_step_reward(reward: torch.Tensor, not_done: torch.Tensor, discount: float):
    ms_reward = 0
    scale = 1
    for i in range(reward.shape[1]):
        ms_reward += scale * reward[:, i]
        scale *= discount * not_done[:, i]

    return ms_reward, scale


class MRQAgent:
    def __init__(
        self,
        obs_shape: Sequence[int],
        action_dim: int,
        max_action: float,
        device: torch.device,
        history: int = 1,
        hp: Dict = {},
    ):
        self.hp = Hyperparameters(**hp)
        set_instance_vars(self.hp, self)
        self.device = device
        self.replay_buffer = buffer.ReplayBuffer(
            obs_shape=obs_shape,
            action_dim=action_dim,
            max_action=max_action,
            pixel_obs=False,
            device=device,
            history=history,
            horizon=max(self.enc_horizon, self.Q_horizon),
            max_size=self.buffer_size,
            batch_size=self.batch_size,
            prioritized=self.prioritized,
            initial_priority=self.min_priority,
        )
        # state_dim will be something like len(robot_state) + vlm_embedding_dim
        self.encoder = Encoder(
            state_dim=obs_shape[0],
            action_dim=action_dim,
            num_bins=self.num_bins,
            zs_dim=self.zs_dim,
            za_dim=self.za_dim,
            zsa_dim=self.zsa_dim,
            hdim=self.enc_hdim,
            activation=self.enc_activation,
        ).to(self.device)
        self.encoder_optimizer = torch.optim.AdamW(
            self.encoder.parameters(), lr=self.enc_lr, weight_decay=self.enc_wd
        )
        self.encoder_target = copy.deepcopy(self.encoder)

        self.policy = Policy(
            action_dim=action_dim,
            gumbel_tau=self.gumbel_tau,
            zs_dim=self.zs_dim,
            hdim=self.policy_hdim,
            activation=self.policy_activation,
        ).to(self.device)
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.policy_lr, weight_decay=self.policy_wd
        )
        self.policy_target = copy.deepcopy(self.policy)

        self.value = Value(
            zsa_dim=self.zsa_dim, hdim=self.value_hdim, activation=self.value_activation
        ).to(self.device)
        self.value_optimizer = torch.optim.AdamW(
            self.value.parameters(), lr=self.value_lr, weight_decay=self.value_wd
        )
        self.value_target = copy.deepcopy(self.value)

        self.two_hot = TwoHot(
            device=self.device,
            lower=self.lower,
            upper=self.upper,
            num_bins=self.num_bins,
        )

        self.state_shape = self.replay_buffer.state_shape
        self.action_dim = action_dim

        self.reward_scale, self.target_reward_scale = (1, 0)
        self.training_step = 0

    def select_action(self, state: np.ndarray, use_exploration: bool = True):
        if self.replay_buffer.size < self.buffer_size_before_training:
            return None  # Random action

        with torch.no_grad():
            state_tensor = torch.tensor(
                state, dtype=torch.float, device=self.device
            ).reshape(-1, *self.state_shape)
            state_embedding = self.encoder.state_embedder(state_tensor)
            action = self.policy.get_action(state_embedding)
            if use_exploration:
                action += torch.randn_like(action) * self.exploration_noise

            return int(action.argmax())

    def train(self):
        if self.replay_buffer.size <= self.buffer_size_before_training:
            return

        self.training_step += 1

        if (self.training_step - 1) % self.target_update_freq == 0:
            self.policy_target.load_state_dict(self.policy.state_dict())
            self.value_target.load_state_dict(self.value.state_dict())
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.target_reward_scale = self.reward_scale
            self.reward_scale = self.replay_buffer.reward_scale()

            for _ in range(self.target_update_freq):
                state, action, next_state, reward, not_done = self.replay_buffer.sample(
                    self.enc_horizon, include_intermediate=True
                )
                self.train_encoder(
                    state,
                    action,
                    next_state,
                    reward,
                    not_done,
                    self.replay_buffer.env_terminates,
                )

        state, action, next_state, reward, not_done = self.replay_buffer.sample(
            self.Q_horizon, include_intermediate=False
        )
        reward, term_discount = multi_step_reward(reward, not_done, self.discount)

        Q, Q_target = self.train_rl(
            state,
            action,
            next_state,
            reward,
            term_discount,
            self.reward_scale,
            self.target_reward_scale,
        )

        if self.prioritized:
            priority = (Q - Q_target.expand(-1, 2)).abs().max(1).values
            priority = priority.clamp(min=self.min_priority).pow(self.alpha)
            self.replay_buffer.update_priority(priority)

    def train_encoder(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        not_done: torch.Tensor,
        env_terminates: bool,
    ):
        with torch.no_grad():
            target_state_embedding = self.encoder_target.state_embedder(
                next_state.reshape(-1, *self.state_shape)
            ).reshape(state.shape[0], -1, self.zs_dim)

        pred_state_embedding = self.encoder.state_embedder(
            state[:, 0]
        )  # produce the first state embedding
        prev_not_done = 1
        encoder_loss = 0

        for i in range(self.enc_horizon):
            pred_done, pred_state_embedding, pred_reward = self.encoder.mdp_predict(
                pred_state_embedding, action[:, i]
            )

            dyn_loss = masked_mse(
                pred_state_embedding, target_state_embedding[:, i], prev_not_done
            )
            reward_loss = (
                self.two_hot.cross_entropy_loss(pred_reward, reward[:, i])
                * prev_not_done
            ).mean()
            done_loss = 0
            if env_terminates:
                done_loss = masked_mse(
                    pred_done, 1.0 - not_done[:, i].reshape(-1, 1), prev_not_done
                )

            encoder_loss += (
                self.dyn_weight * dyn_loss
                + self.reward_weight * reward_loss
                + self.done_weight * done_loss
            )
            prev_not_done = not_done[:, i].reshape(-1, 1) * prev_not_done

        self.encoder_optimizer.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_optimizer.step()

    def train_rl(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
        term_discount: torch.Tensor,
        reward_scale: float,
        target_reward_scale: float,
    ):
        with torch.no_grad():
            next_state_embedding = self.encoder_target.state_embedder(next_state)

            noise = (torch.randn_like(action) * self.target_policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = self.policy_target.get_action(next_state_embedding) + noise
            next_action = F.one_hot(next_action.argmax(1), next_action.shape[1]).float()

            next_state_action_embedding = self.encoder_target(
                next_state_embedding, next_action
            )
            Q_target = (
                self.value_target(next_state_action_embedding)
                .min(1, keepdim=True)
                .values
            )
            Q_target = (
                reward + term_discount * Q_target * target_reward_scale
            ) / reward_scale

            state_embedding = self.encoder.state_embedder(state)
            state_action_embedding = self.encoder(state_embedding, action)

        Q = self.value(state_action_embedding)
        value_loss = F.smooth_l1_loss(Q, Q_target.expand(-1, 2))

        self.value_optimizer.zero_grad(set_to_none=True)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.value_grad_clip)
        self.value_optimizer.step()

        policy_action, policy_action_embedding = self.policy(state_embedding)
        state_action_embedding = self.encoder(state_embedding, policy_action)
        Q_policy_action = self.value(state_action_embedding)
        policy_loss = (
            -Q_policy_action.mean()
            + self.policy_action_weight * policy_action_embedding.pow(2).mean()
        )

        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        self.policy_optimizer.step()

        return Q, Q_target

    def save(self, save_folder: str):
        # Save models/optimizers
        models = [
            "encoder",
            "encoder_target",
            "encoder_optimizer",
            "policy",
            "policy_target",
            "policy_optimizer",
            "value",
            "value_target",
            "value_optimizer",
        ]
        for k in models:
            torch.save(self.__dict__[k].state_dict(), f"{save_folder}/{k}.pt")

        # Save variables
        vars = ["hp", "reward_scale", "target_reward_scale", "training_steps"]
        var_dict = {k: self.__dict__[k] for k in vars}
        np.save(f"{save_folder}/agent_var.npy", var_dict)

        self.replay_buffer.save(save_folder)

    def load(self, save_folder: str):
        # Load models/optimizers.
        models = [
            "encoder",
            "encoder_target",
            "encoder_optimizer",
            "policy",
            "policy_target",
            "policy_optimizer",
            "value",
            "value_target",
            "value_optimizer",
        ]
        for k in models:
            self.__dict__[k].load_state_dict(
                torch.load(f"{save_folder}/{k}.pt", weights_only=True)
            )

        # Load variables.
        var_dict = np.load(f"{save_folder}/agent_var.npy", allow_pickle=True).item()
        for k, v in var_dict.items():
            self.__dict__[k] = v

        self.replay_buffer.load(save_folder)


def main():
    agent = MRQAgent(
        obs_shape=(3, 64, 64),
        action_dim=5,
        max_action=1.0,
        device=torch.device("cuda"),
        history=2,
    )


if __name__ == "__main__":
    main()
