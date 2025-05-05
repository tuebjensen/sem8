from functools import partial
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def ln_activation(x: torch.Tensor, activation: Callable) -> torch.Tensor:
    x = F.layer_norm(x, (x.shape[-1],))
    return activation(x)


def weight_init(layer):
    if isinstance(layer, nn.Linear):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(layer.weight.data, gain)
        if hasattr(layer.bias, "data"):
            layer.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Sequence[int] | int,
        output_dim: int,
        activiation: Callable = F.elu,
    ) -> None:
        super().__init__()
        layers = []
        if isinstance(hidden_dim, int):
            layers.append(nn.Linear(input_dim, hidden_dim))
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            layers.append(nn.Linear(input_dim, hidden_dim[0]))
            for i in range(1, len(hidden_dim)):
                layers.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))
            self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

        self.hidden_model = nn.Sequential(*layers)
        self.activiation = activiation
        self.apply(weight_init)

    def forward(self, x):
        for layer in self.hidden_model:
            x = layer(x)
            x = ln_activation(x, self.activiation)
        return self.output_layer(x)


class Encoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_bins: int = 65,
        zs_dim: int = 512,
        za_dim: int = 256,
        zsa_dim: int = 512,
        hdim: int = 512,
        activation: Callable = F.elu,
    ):
        super().__init__()
        self.state_embedder = MLP(state_dim, hdim, zs_dim, activation)
        self.action_embedder = nn.Linear(action_dim, za_dim)

        self.state_action_embedder = MLP(zs_dim + za_dim, hdim, zsa_dim, activation)
        self.state_action_predictor = nn.Linear(
            zsa_dim, num_bins + zs_dim + 1
        )  # num_bins for reward, zs_dim for next state, 1 for done

        self.zs_dim = zs_dim
        self.actication = activation
        self.apply(weight_init)

    def forward(self, state_embedding: torch.Tensor, action: torch.Tensor):
        action_embedding = self.action_embedder(action)
        return self.state_action_embedder(
            torch.cat([state_embedding, action_embedding], 1)
        )

    def mdp_predict(self, state_embedding: torch.Tensor, action: torch.Tensor):
        state_action_embedding = self.forward(state_embedding, action)
        dzr = self.state_action_predictor(state_action_embedding)
        return (
            dzr[:, 0:1],
            dzr[:, 1 : self.zs_dim + 1],
            dzr[:, self.zs_dim + 1 :],
        )  # done, next_state, reward


class Policy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        gumbel_tau: float = 10,
        zs_dim: int = 512,
        hdim: int = 512,
        activation: Callable = F.relu,
    ):
        super().__init__()
        self.policy = MLP(zs_dim, hdim, action_dim, activation)
        self.activation = partial(F.gumbel_softmax, tau=gumbel_tau)

    def forward(self, state_embedding: torch.Tensor):
        action_embedding = self.policy(state_embedding)
        action = self.activation(action_embedding)
        return action, action_embedding

    def get_action(self, state_embedding: torch.Tensor):
        action, _ = self.forward(state_embedding)
        return action


class Value(nn.Module):
    def __init__(self, zsa_dim: int, hdim: int = 512, activation: Callable = F.elu):
        super().__init__()
        self.q1 = MLP(zsa_dim, [hdim, hdim], 1, activation)
        self.q2 = MLP(zsa_dim, [hdim, hdim], 1, activation)

        self.activation = activation

    def forward(self, state_action_embedding: torch.Tensor):
        q1 = self.q1(state_action_embedding)
        q2 = self.q2(state_action_embedding)
        return torch.cat([q1, q2], 1)
