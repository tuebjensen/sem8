import gymnasium as gym
import Sem8Env as _

env = gym.make("Sem8-v0", render_mode="human")
observation, info = env.reset()

episode_over = False
episode_count = 10

for i in range(episode_count):
    print(f"Episode {i + 1} of {episode_count}")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = (
            env.action_space.sample()
        )  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

env.close()
