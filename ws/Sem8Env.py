import json
import math
import os
import random
import sys
import timeit
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from mazelib import Maze

from create_dataset import generate_image, generate_maze


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
        flattened_mask = mask.T.flatten() if mask is not None else None
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
        self._agent_speed = 10
        self._agent_turn_speed = 15
        self._agent_radius = 15
        self._eval = kwargs.get("eval", False)
        self._eval_data_dir = kwargs.get("eval_data_dir", "")
        self.is_last_eval_data = False
        self._eval_data_generator = self._load_eval_data()

        self.action_space = spaces.Discrete(4)  # Forward, Left, Right, Pick up
        self.observation_space = spaces.Tuple(
            (
                DiscreteBoxSpace(1, 1),  # agent position
                spaces.Discrete(1),  # angle
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
        self._maze_generator = Maze()

    def _get_obs(self):
        return (
            np.floor(self._agent_position),
            np.floor(self._agent_angle),
        )

    def _get_info(self):
        return {
            "image": self._image,
        }

    def _load_random_image_bbox(self):
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

        return image, bboxes, category_ids

    def _apply_boundary_mask(self, mask: np.ndarray, radius: int):
        mask[0 : radius + 1] = 0
        mask[-radius:] = 0
        mask[:, 0 : radius + 1] = 0
        mask[:, -radius:] = 0
        return mask

    def _apply_rect_mask(self, mask: np.ndarray, rects: list[pygame.Rect], radius: int):
        modified_mask = mask.copy()
        inflated_radius = round(radius * 1.2)
        for rect in rects:
            left = max(0, rect.left - inflated_radius)
            right = min(self._width, rect.right + inflated_radius)
            top = max(0, rect.top - inflated_radius)
            bottom = min(self._height, rect.bottom + inflated_radius)
            modified_mask[left:right, top:bottom] = 0
        return modified_mask

    def _make_bbox_rects(self, bboxes):
        bbox_rects = []
        for bbox in bboxes:
            rect = pygame.Rect(bbox)
            bbox_rects.append(rect)
        return bbox_rects

    def _draw_rects(
        self,
        surface: pygame.Surface,
        rects: list[pygame.Rect],
        color=(0, 255, 0),
        line_width=0,
    ):
        # Returns new surface with maze rects drawn on it
        surface_with_rects = surface.copy()
        for rect in rects:
            pygame.draw.rect(surface_with_rects, color, rect, line_width)
        return surface_with_rects

    def _generate_train_data(self):
        self._image, self._bbox_rects, self._categories = generate_image()
        self._width, self._height = self._image.get_size()

        self._target_bbox_index = random.randint(0, len(self._bbox_rects) - 1)

        self.observation_space = spaces.Tuple(
            (
                DiscreteBoxSpace(self._width, self._height),  # agent position
                spaces.Discrete(360),  # angle
            )
        )
        agent_mask = np.ones((self._width, self._height), dtype=np.int8)
        angle_mask = np.ones(360, dtype=np.int8)
        agent_mask = self._apply_boundary_mask(agent_mask, self._agent_radius)
        agent_mask = self._apply_rect_mask(
            agent_mask, self._bbox_rects, self._agent_radius
        )
        state_sample = self.observation_space.sample(mask=(agent_mask, angle_mask))
        self._agent_position, self._agent_angle = state_sample
        self._agent_position = np.array(self._agent_position, dtype=np.float64)

        self._image, self._maze_rects, self._bbox_rects = generate_maze(
            self._maze_generator,
            self._image,
            self._bbox_rects,
            self._agent_position,
            self._agent_radius,
        )
        self._prompt = "You are guiding a robot to pick up an object in a maze. "
        self._prompt += "The robot is represented by a green circle with red line indicating the direction it's pointing in. "
        self._prompt += (
            "The maze is represented by either horizontal or vertical green lines. "
        )
        self._prompt += (
            "The robot can move forward, turn left or right, and pick up the object. "
        )
        self._prompt += f"Your goal is to locate the {self._categories[self._target_bbox_index]} without hitting the maze."

    def _load_eval_data(self):
        print("Loading eval data from", self._eval_data_dir)
        with open(os.path.join(self._eval_data_dir, "meta_data.json")) as f:
            eval_meta_data = json.load(f)
        while True:
            for image_data in eval_meta_data:
                image_name = image_data["image_file_name"]
                image_path = os.path.join(self._eval_data_dir, image_name)
                self._image = pygame.image.load(image_path)
                self._width, self._height = self._image.get_size()
                self.observation_space = spaces.Tuple(
                    (
                        DiscreteBoxSpace(self._width, self._height),  # agent position
                        spaces.Discrete(360),  # angle
                    )
                )

                self._bbox_rects = [pygame.Rect(bbox) for bbox in image_data["bboxes"]]
                self._target_bbox_index = image_data["target_bbox_index"]
                self._categories = image_data["categories"]
                self._maze_rects = [
                    pygame.Rect(maze_xywh) for maze_xywh in image_data["maze_xywh"]
                ]

                self._agent_position = np.array(
                    image_data["agent_position"], dtype=np.float64
                )
                self._agent_angle = image_data["agent_angle"]
                self._prompt = image_data["prompt"]
                print(self._prompt)
                # print("Goal is", self._categories[self._target_bbox_index])

                yield

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        if self._eval:
            try:
                next(self._eval_data_generator)
            except StopIteration:
                self.is_last_eval_data = True
        else:
            self._generate_train_data()
        # self._draw_rects(self._image, self._bbox_rects, color=(255, 0, 0), line_width=2)
        self._image_with_maze = self._draw_rects(
            self._image, self._maze_rects, color=(0, 255, 0)
        )
        # Sample returns integers, but we want floats for math purposes
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

    def step(self, action):
        reward = -1
        terminated = False
        if action == 0:
            # Go forward
            new_x = self._agent_position[0] + self._agent_speed * np.cos(
                np.radians(self._agent_angle)
            )
            new_y = self._agent_position[1] + self._agent_speed * np.sin(
                np.radians(self._agent_angle)
            )

            # Check for collision with maze
            maze_collided = True
            for rect in self._maze_rects:
                maze_collided, coll_x, coll_y = self._circle_aa_rect_collision(
                    (new_x, new_y),
                    self._agent_radius,
                    rect,
                )
                if maze_collided:
                    # pygame.draw.circle(
                    #     self._image_with_maze,
                    #     (0, 0, 255),
                    #     (coll_x, coll_y),
                    #     3,
                    # )
                    break

            if not maze_collided:
                self._agent_position[0] = new_x
                self._agent_position[1] = new_y

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
            correct_pick_up, coll_x, coll_y = self._circle_aa_rect_collision(
                self._agent_position,
                self._agent_radius,
                self._bbox_rects[self._target_bbox_index],
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
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # render_surface = pygame.Surface((self._width, self._height))
        render_surface = self._image_with_maze.copy()
        # render_surface.blit(self._image_with_maze, (0, 0))
        pygame.draw.circle(
            render_surface,
            (0, 255, 0),
            self._agent_position,
            self._agent_radius,
        )
        pygame.draw.line(
            render_surface,
            (255, 0, 0),
            self._agent_position,
            (
                self._agent_position[0] + 25 * np.cos(np.radians(self._agent_angle)),
                self._agent_position[1] + 25 * np.sin(np.radians(self._agent_angle)),
            ),
            4,
        )

        if self.render_mode == "human":
            assert self.window is not None
            assert self.clock is not None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

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


def main():
    env = gym.make("Sem8-v0", render_mode="human", eval=True, eval_data_dir="test")

    for i in range(10):
        observation, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # print(observation, reward, terminated, truncated, info)
    env.close()


if __name__ == "__main__":
    main()
