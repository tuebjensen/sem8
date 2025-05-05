import json
import os
from typing import NamedTuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

# from MRQ.MRQAgent import MRQAgent
from original_MRQ.MRQ import Agent


class Params(NamedTuple):
    total_episodes: int  # Total episodes
    learning_rate: float  # Learning rate
    gamma: float  # Discounting rate
    epsilon: float  # Exploration probability
    map_size: int  # Number of tiles of one side of the squared environment
    seed: int  # Define a seed so that we get reproducible results
    is_slippery: bool  # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int  # Number of runs
    action_size: int  # Number of possible actions
    state_size: int  # Number of possible states
    proba_frozen: float  # Probability that a tile is frozen
    savefig_folder: str  # Root folder where plots are saved


class ExperimentRunner:
    def __init__(self, agent: Agent, env, params: Params):
        self.agent = agent
        self.env = env
        self.total_rewards = []
        self.params = params

    def run(self, episodes):
        for episode in tqdm(range(episodes)):
            total_reward = 0
            state, _ = self.env.reset(seed=self.params.seed)
            done = False
            while not done:
                state = np.array([state])
                action = self.agent.select_action(state)
                if action is None:
                    action = self.env.action_space.sample()
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.replay_buffer.add(
                    state, action, np.array([next_state]), reward, terminated, truncated
                )

                state = next_state
                total_reward += reward
                self.agent.train()
            self.total_rewards.append(total_reward)

    def save_results(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.agent.save(save_dir)
        results = {
            "total_rewards": self.total_rewards,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f)


def main():
    # Create the environment
    params = Params(
        total_episodes=1000,
        learning_rate=0.8,
        gamma=0.95,
        epsilon=0.1,
        map_size=3,
        seed=123,
        is_slippery=False,
        n_runs=20,
        action_size=None,
        state_size=None,
        proba_frozen=0.9,
        savefig_folder="./results/frozen_lake_1",
    )
    desc = generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    )
    env = gym.make(
        "FrozenLake-v1",
        is_slippery=params.is_slippery,
        render_mode="rgb_array",
        desc=desc,
    )
    env = TimeLimit(env, max_episode_steps=100)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # agent = MRQAgent((1,), env.action_space.n, 1, device, 1)
    agent = Agent((1,), env.action_space.n, 1, False, True, device, 1)
    experiment_runner = ExperimentRunner(agent, env, params)
    experiment_runner.run(params.total_episodes)
    experiment_runner.save_results(params.savefig_folder)
    # for _ in range(100):
    #     state, _ = env.reset(seed=params.seed)
    #     state = np.array([state])

    #     action = agent.select_action(state, use_exploration=False)
    #     if action is None:
    #         action = env.action_space.sample()
    #     next_state, reward, terminated, truncated, _ = env.step(action)
    #     env.render()


def modify_parser(parser):
    pass


if __name__ == "__main__":
    main()
