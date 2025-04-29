import json
import math
import os
import random
import sys
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


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
        self._annotations_path = os.path.join("output", "meta_data.json")
        with open(self._annotations_path) as f:
            self._annotation_dict = json.load(f)
        self._categories_dict = {
            category["id"]: category for category in self._annotation_dict["categories"]
        }

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

    def _load_random_image_bbox_maze(self):
        image_object = random.choice(self._annotation_dict["images"])
        image_file_name = image_object["file_name"]
        image_dir = os.path.dirname(self._annotations_path)
        image_path = os.path.join(image_dir, image_file_name)
        image = pygame.image.load(image_path)

        bboxes = []
        category_ids = []
        for bbox_object in image_object["bbox_objects"]:
            bbox = bbox_object["bbox"]
            category_id = bbox_object["category_id"]
            bboxes.append(bbox)
            category_ids.append(category_id)
        maze_lines = random.choice(image_object["mazes"])

        return image, bboxes, category_ids, maze_lines

    def _apply_boundary_mask(self, mask: np.ndarray, radius: int):
        mask[0 : radius + 1] = 0
        mask[-radius:] = 0
        mask[:, 0 : radius + 1] = 0
        mask[:, -radius:] = 0
        return mask

    def _apply_rect_mask(self, mask: np.ndarray, rects: list[pygame.Rect]):
        for rect in rects:
            mask[rect.left : rect.right, rect.top : rect.bottom] = 0
        return mask

    def _make_bbox_rects(self, bboxes):
        bbox_rects = []
        for bbox in bboxes:
            rect = pygame.Rect(bbox)
        return bbox_rects

    def _make_maze_rects(self, maze_lines):
        maze_rects = []
        for line in maze_lines:
            line_width = 2
            x_start = math.floor(line["line_start"][0])
            y_start = math.floor(line["line_start"][1])
            x_end = math.floor(line["line_end"][0])
            y_end = math.floor(line["line_end"][1])
            rect_width = max(abs(x_end - x_start), line_width)
            rect_height = max(abs(y_end - y_start), line_width)
            x_start = min(x_start, self._width - line_width)
            y_start = min(y_start, self._height - line_width)
            maze_rect = pygame.Rect(x_start, y_start, rect_width, rect_height)
            maze_rects.append(maze_rect)
        return maze_rects

    def _draw_rects(self, surface, rects):
        for rect in rects:
            pygame.draw.rect(surface, (0, 255, 0), rect)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        self._image, self._bboxes, self._category_ids, self._maze_lines = (
            self._load_random_image_bbox_maze()
        )
        self._maze_rects = self._make_maze_rects(self._maze_lines)
        self._bbox_rects = self._make_bbox_rects(self._bboxes)
        self._draw_rects(self._image, self._maze_rects)
        self._width, self._height = self._image.get_size()
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
        agent_mask = self._apply_rect_mask(
            agent_mask, self._bbox_rects, self._agent_radius
        )
        target_mask = self._apply_boundary_mask(target_mask, self._target_radius)
        target_mask = self._apply_rect_mask(
            target_mask,
            self._bbox_rects,
        )

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

    def _circle_aa_rect_collision(self, center, radius, rect):
        # collision between circle and axis-aligned rectangle
        coll_x, coll_y = center
        if coll_x < rect.left:
            coll_x = rect.left
        elif coll_x > rect.right:
            coll_x = rect.right
        if coll_y < rect.top:
            coll_y = rect.top
        elif coll_y > rect.bottom:
            coll_y = rect.bottom
        dist_x = center[0] - coll_x
        dist_y = center[1] - coll_y
        distance_squared = dist_x * dist_x + dist_y * dist_y
        collided = distance_squared <= radius * radius
        return collided, coll_x, coll_y

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

        # render_surface = pygame.Surface((self._width, self._height))
        render_surface = self._image
        # render_surface.blit(self._image, (0, 0))
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
