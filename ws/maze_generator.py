import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pygame
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms


# load image with bounding box
def load_random_image_with_bbox():
    annotation_file = "images/instances_default.json"
    with open(annotation_file) as f:
        annotation_dict = json.load(f)
    image_object = random.choice(annotation_dict["images"])
    image_file_name = image_object["file_name"]
    image_dir = "images"
    image_path = os.path.join(image_dir, image_file_name)
    image = pygame.image.load(image_path)
    image_id = image_object["id"]
    annotations = annotation_dict["annotations"]
    bboxes = []
    for annotation in annotations:
        if annotation["image_id"] == image_id:
            bbox = annotation["bbox"]
            bboxes.append(bbox)
    return image, bboxes


def scale_bboxes(bboxes, x_scale, y_scale):
    scaled_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        scaled_bboxes.append(
            (
                round(x * x_scale),
                round(y * y_scale),
                round(w * x_scale),
                round(h * y_scale),
            )
        )
    return scaled_bboxes


# draw bboxes
def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        x, y, w, h = bbox
        pygame.draw.rect(image, (255, 0, 0), (x, y, w, h), 2)
    return image


def draw_lines(surface, lines):
    for line in lines:
        pygame.draw.line(surface, (0, 255, 0), line["line_start"], line["line_end"], 3)


def draw_maze(surface, grid, cell_size):
    # We have to divide the coordinates by 2 because mazelib generates a bigger grid
    lines = lines_from_maze(grid, cell_size)
    draw_lines(surface, lines)


def lines_from_maze(grid, cell_size):
    lines = []
    # We have to divide the coordinates by 2 because mazelib generates a bigger grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j] == 1:
                if i > 0 and grid[i - 1, j] == 1:
                    line_start = (j / cell_size / 2, i / cell_size / 2)
                    line_end = (j / cell_size / 2, (i - 1) / cell_size / 2)
                    # Flip line start and end to make the line go from top to bottom, left to right
                    lines.append({"line_start": line_end, "line_end": line_start})

                if j > 0 and grid[i, j - 1] == 1:
                    line_start = (j / cell_size / 2, i / cell_size / 2)
                    line_end = ((j - 1) / cell_size / 2, i / cell_size / 2)
                    lines.append({"line_start": line_end, "line_end": line_start})
    return lines


def rooms_from_bboxes(bboxes, cell_size):
    # Calculate the room coordinates in the grid
    # The coordinates are in the format (y, x) for the grid
    # Multiply by 2 for similar reasons as in draw_maze
    rooms = []
    for bbox in bboxes:
        x, y, w, h = bbox
        top_left_coord = (math.floor(y * cell_size * 2), math.floor(x * cell_size * 2))
        bottom_right_coord = (
            math.floor((y + h) * cell_size * 2),
            math.floor((x + w) * cell_size * 2),
        )
        room = [top_left_coord, bottom_right_coord]
        rooms.append(room)
    return rooms


def main():
    pygame.init()
    image, bboxes = load_random_image_with_bbox()
    width, height = image.get_size()
    desired_grid_width = 5
    desired_grid_height = 10
    # Maze density
    desired_grid_area = desired_grid_width * desired_grid_height
    image_area = width * height
    scale = math.sqrt(desired_grid_area / image_area)
    grid_width = round(width * scale)
    grid_height = round(height * scale)
    new_image_width = grid_width / scale
    new_image_height = grid_height / scale
    image = pygame.transform.scale(image, (new_image_width, new_image_height))
    bbox_x_scale = new_image_width / width
    bbox_y_scale = new_image_height / height
    bboxes = scale_bboxes(bboxes, bbox_x_scale, bbox_y_scale)
    window = pygame.display.set_mode((new_image_width * 2, new_image_height * 2))
    image = draw_bboxes(image, bboxes)
    rooms = rooms_from_bboxes(bboxes, scale)
    m = Maze()
    m.generator = DungeonRooms(grid_height, grid_width, rooms=rooms)

    m.generate()
    grid = m.grid
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.blit(image, (0, 0))
        draw_maze(window, grid, scale)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
