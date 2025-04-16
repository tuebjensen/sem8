import sys
import math
from typing import Any
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import json
import os
import random


class DiscreteBoxSpace(spaces.Space):
    def __init__(self, width, height, shape=None, dtype=np.int64):
        super().__init__(shape=shape, dtype=dtype)
        self._width = width
        self._height = height

        self._space = spaces.Discrete(width * height)

    # mask is a 2D numpy array of np.int8
    def sample(
        self, mask: np.ndarray | None = None, probability: np.ndarray | None = None
    ):
        flattened_mask = mask.flatten() if mask is not None else None
        position = self._space.sample(mask=flattened_mask, probability=probability)
        x, y = self._index_to_xy(position)

        return np.array([x, y], dtype=self.dtype)

    def _index_to_xy(self, index):
        x = index % self._width
        y = index // self._width
        return x, y

    def contains(self, x) -> bool:
        x_coordinate, y_coordinate = x
        return 0 <= x_coordinate < self._width and 0 <= y_coordinate < self._height


class Sem8Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, **kwargs) -> None:
        super().__init__()
        self._agent_angle = 0.0
        self._agent_speed = 25
        self._agent_turn_speed = 5
        self._agent_radius = 60

        self._target_radius = 60

        self.action_space = spaces.Discrete(4)  # Forward, Left, Right, Pick up
        self.observation_space = spaces.Tuple(
            (
                DiscreteBoxSpace(1, 1),  # agent position
                spaces.Discrete(1),  # angle
                DiscreteBoxSpace(1, 1),  # target position
            )
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        with open("images/instances_default.json") as f:
            self._annotation_dict = json.load(f)

    def _get_obs(self):
        return (
            np.floor(self._agent_position),
            np.floor(self._agent_angle),
            np.floor(self._target_position),
        )

    def _get_info(self):
        return {
            "image": self._image,
        }

    def _load_random_image_with_bbox(self):
        image_object = random.choice(self._annotation_dict["images"])
        image_file_name = image_object["file_name"]
        image_dir = "images"
        image_path = os.path.join(image_dir, image_file_name)
        image = pygame.image.load(image_path)
        image_id = image_object["id"]
        annotations = self._annotation_dict["annotations"]
        bboxes = []
        for annotation in annotations:
            if annotation["image_id"] == image_id:
                bbox = annotation["bbox"]
                bboxes.append(bbox)
        return image, bboxes

    def _apply_boundary_mask(self, mask: np.ndarray, radius: int):
        mask[0 : radius + 1] = 0
        mask[-radius:] = 0
        mask[:, 0 : radius + 1] = 0
        mask[:, -radius:] = 0
        return mask

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._image, self._bboxes = self._load_random_image_with_bbox()
        self._width, self._height = self._image.get_width(), self._image.get_height()
        self.observation_space = spaces.Tuple(
            (
                DiscreteBoxSpace(self._width, self._height),  # agent position
                spaces.Discrete(360),  # angle
                DiscreteBoxSpace(self._width, self._height),  # target position
            )
        )
        agent_mask = np.ones((self._width, self._height), dtype=np.int8)
        angle_mask = np.ones(360, dtype=np.int8)
        target_mask = np.ones((self._width, self._height), dtype=np.int8)

        agent_mask = self._apply_boundary_mask(agent_mask, self._agent_radius)
        target_mask = self._apply_boundary_mask(target_mask, self._target_radius)

        state_sample = self.observation_space.sample(
            mask=(agent_mask, angle_mask, target_mask)
        )
        self._agent_position, self._agent_angle, self._target_position = state_sample
        # Sample returns integers, but we want floats for math purposes
        self._agent_position = np.array(self._agent_position, dtype=np.float64)
        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        return_info = self._get_info()

        return observation, return_info

    def _agent_collide_with_object(self, agent_pos: np.ndarray, object_pos: np.ndarray):
        distance = np.linalg.norm(agent_pos - object_pos)
        return distance <= self._target_radius + self._agent_radius

    def step(self, action):
        reward = -1
        terminated = False
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
        elif action == 1:
            self._agent_angle = (self._agent_angle + self._agent_turn_speed) % 360
        elif action == 2:
            self._agent_angle = (self._agent_angle - self._agent_turn_speed) % 360
        elif action == 3:
            correct_pick_up = self._agent_collide_with_object(
                self._agent_position, self._target_position
            )
            reward = 5 if correct_pick_up else -5
            terminated = correct_pick_up

        if self.render_mode == "human":
            self._render_frame()

        truncated = False
        observation = self._get_obs()
        return_info = self._get_info()

        return observation, reward, terminated, truncated, return_info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

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
            self._agent_position,
            self._agent_radius,
        )
        pygame.draw.circle(
            render_surface,
            (0, 0, 255),
            self._target_position,
            self._target_radius,
        )
        pygame.draw.line(
            render_surface,
            (255, 0, 0),
            self._agent_position,
            (
                self._agent_position[0] + 50 * np.cos(np.radians(self._agent_angle)),
                self._agent_position[1] + 50 * np.sin(np.radians(self._agent_angle)),
            ),
            4,
        )

        if self.render_mode == "human":
            assert self.window is not None
            assert self.clock is not None
            for bbox in self._bboxes:
                x, y, w, h = bbox
                pygame.draw.rect(
                    render_surface,
                    (255, 0, 0),
                    (x, y, w, h),
                    3,
                )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # Draw image bounding boxes

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
