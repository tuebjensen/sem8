import json
import math
import os
import random

import pygame
from mazelib import Maze

from create_dataset import generate_image, generate_maze


def draw_lines(surface, lines):
    for line in lines:
        pygame.draw.line(surface, (0, 255, 0), line[0], line[1], 3)


def draw_rects(surface, rects):
    for rect in rects:
        pygame.draw.rect(surface, (255, 0, 0), rect, 2)


def main():
    pygame.init()
    image, bboxes, categories = generate_image()
    maze_object = Maze()
    image, maze_lines, bboxes = generate_maze(
        maze_object,
        image,
        bboxes,
        spawn_point=(0, 0),
        agent_radius=10,
    )
    width, height = image.get_size()
    window = pygame.display.set_mode((width, height))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.blit(image, (0, 0))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
