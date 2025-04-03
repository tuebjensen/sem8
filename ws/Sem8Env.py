from math import floor
import math
from typing import Any, override
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame


class Sem8Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, **kwargs) -> None:
        super().__init__()
        self._agent_position = np.array([-1, -1], dtype=np.float32)
        self._agent_angle = 0.0
        self._agent_speed = 5
        self._agent_turn_speed = 0.5
        self._agent_radius = 30

        self._target_radius = 30

        self.action_space = spaces.Discrete(3)  # Forward, Left, Right
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(1),  # x coordinate
                spaces.Discrete(1),  # y coordinate
                spaces.Discrete(1),  # angle
            )
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return (
            math.floor(self._agent_position[0]),
            math.floor(self._agent_position[1]),
            math.floor(self._agent_angle),
        )

    def _get_info(self):
        return {
            "image": self._image,
        }

    @override
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        image_path = "images/background.png"
        self._image = pygame.image.load(image_path)
        self._width, self._height = self._image.get_width(), self._image.get_height()
        # im_frame = Image.open(image_path)
        # self._image = np.array(im_frame.getdata())
        # self._width, self._height = self._image.shape
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(self._width),  # x coordinate
                spaces.Discrete(self._height),  # y coordinate
                spaces.Discrete(360),  # angle
            )
        )
        x_mask = np.ones(self._width, dtype=np.int8)
        y_mask = np.ones(self._height, dtype=np.int8)
        angle_mask = np.ones(360, dtype=np.int8)
        x_mask[0 : self._agent_radius + 1] = 0
        x_mask[self._width - self._agent_radius :] = 0
        y_mask[0 : self._agent_radius + 1] = 0
        y_mask[self._height - self._agent_radius :] = 0
        agent_init = self.observation_space.sample(mask=(x_mask, y_mask, angle_mask))
        self._agent_position[0], self._agent_position[1], self._agent_angle = agent_init
        x_mask = np.ones(self._width, dtype=np.int8)
        y_mask = np.ones(self._height, dtype=np.int8)
        x_mask[0 : self._target_radius + 1] = 0
        x_mask[self._width - self._target_radius :] = 0
        y_mask[0 : self._target_radius + 1] = 0
        y_mask[self._height - self._target_radius :] = 0
        target_init = self.observation_space.sample(mask=(x_mask, y_mask, angle_mask))
        self._target_position = np.array([target_init[0], target_init[1]])

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        return_info = self._get_info()

        return observation, return_info

    def _agent_collide_with_object(self, agent_pos: np.ndarray, object_pos: np.ndarray):
        distance = np.linalg.norm(agent_pos - object_pos)
        return distance <= self._target_radius + self._agent_radius

    @override
    def step(self, action):
        if action == 0:
            # Go forward
            self._agent_position[0] += self._agent_speed * np.cos(
                np.radians(self._agent_angle)
            )
            self._agent_position[1] += self._agent_speed * np.sin(
                np.radians(self._agent_angle)
            )
            self._agent_position[0:2] = np.clip(
                self._agent_position[0:2],
                [self._agent_radius, self._agent_radius],
                [self._width - self._agent_radius, self._height - self._agent_radius],
            )
        if action == 1:
            self._agent_angle += self._agent_turn_speed
        if action == 2:
            self._agent_angle -= self._agent_turn_speed

        if self.render_mode == "human":
            self._render_frame()

        terminated = self._agent_collide_with_object(
            self._agent_position, self._target_position
        )
        truncated = False
        reward = 1 if terminated else -1
        observation = self._get_obs()
        return_info = self._get_info()

        return observation, reward, terminated, truncated, return_info

    @override
    def render(self):
        if self.render_mode == "rgb_array":
            self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self._width, self._height))
        if self.clock == None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        render_surface = pygame.Surface((self._width, self._height))
        render_surface.blit(self._image, (0, 0))

        pygame.draw.circle(
            render_surface,
            (0, 255, 0),
            self._agent_position.tolist(),
            self._agent_radius,
        )
        pygame.draw.circle(
            render_surface,
            (0, 0, 255),
            self._target_position.tolist(),
            self._target_radius,
        )
        pygame.draw.line(
            render_surface,
            (255, 0, 0),
            self._agent_position.tolist(),
            (
                self._agent_position[0] + 50 * np.cos(np.radians(self._agent_angle)),
                self._agent_position[1] + 50 * np.sin(np.radians(self._agent_angle)),
            ),
            4,
        )

        if self.render_mode == "human":
            assert self.window is not None
            assert self.clock is not None
            self.window.blit(render_surface, render_surface.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(render_surface)), axes=(1, 0, 2)
            )

        def close(self):
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()


gym.register(id="Sem8-v0", entry_point=Sem8Env)
