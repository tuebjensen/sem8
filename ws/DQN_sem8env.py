import gymnasium as gym
from DQN import DQNAgent, plot_durations
import torch
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
import Sem8Env as _
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import logging
import numpy as np
from typing import Iterable
from tqdm import tqdm

is_ipython = "inline" in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def flatten_state(state, device=None):
    flattened_state = []

    def _flatten_state(state):
        # Flatten the state tuple into a single tensor
        for s in state:
            if isinstance(s, Iterable):
                _flatten_state(s)
            else:
                flattened_state.append(s)

    _flatten_state(state)
    flattened_state = np.array(flattened_state, dtype=np.float32)
    return torch.tensor(flattened_state, dtype=torch.float32, device=device)


def main():
    episode_durations = []
    env = gym.make("Sem8-v0", render_mode="rgb_array")
    state, info = env.reset()
    state = flatten_state(state)
    dqn_agent = DQNAgent(
        len(state), env.action_space.n, env.action_space, hidden_dim=128
    )
    dqn_agent.batch_size = 32
    device = dqn_agent.device
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 6000
    else:
        num_episodes = 1000

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = flatten_state(state, device).unsqueeze(0)
        for t in count():
            action = dqn_agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                # Flatten the next state
                next_state = flatten_state(observation, device).unsqueeze(0)

            # Store the transition in memory
            dqn_agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            dqn_agent.optimize_model()

            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    print("Complete")
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
