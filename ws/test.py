import argparse
import gymnasium as gym
import Sem8Env as _
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
import logging


def main():
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    env = gym.make("Sem8-v0", render_mode="rgb_array")
    observation, info = env.reset()

    episode_over = False
    episode_count = 10
    training_period = 2

    env = RecordVideo(
        env,
        video_folder="videos",
        name_prefix="eval",
        episode_trigger=lambda x: x % training_period == 0,
    )
    env = RecordEpisodeStatistics(env)

    for i in range(episode_count):
        observation, info = env.reset()

        episode_over = False
        while not episode_over:
            action = (
                env.action_space.sample()
            )  # agent policy that uses the observation and info
            observation, reward, terminated, truncated, info = env.step(action)

            episode_over = terminated or truncated

        info["episode"]["i"] = i
        logger.info(info["episode"])

    env.close()


def modify_parser(parser: argparse.ArgumentParser):
    pass


if __name__ == "__main__":
    main()
