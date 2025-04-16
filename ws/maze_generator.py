import numpy as np
import os
import pygame
import random
import json
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
import math
import matplotlib.pyplot as plt

annotation_file = "images/instances_default.json"

with open(annotation_file) as f:
    annotation_dict = json.load(f)


# load image with bounding box
def load_random_image_with_bbox():
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


def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation="nearest")
    plt.xticks([]), plt.yticks([])
    plt.show()


def draw_line(image, line_start, line_end):
    pygame.draw.line(image, (0, 255, 0), line_start, line_end, 3)


def main():
    pygame.init()
    image, bboxes = load_random_image_with_bbox()
    width, height = image.get_size()
    desired_grid_width = 5
    desired_grid_height = 10
    desired_grid_area = (
        desired_grid_width * desired_grid_height
    )  # desired_grid_width * desired_grid_height
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
    window = pygame.display.set_mode((new_image_width, new_image_height))
    image = draw_bboxes(image, bboxes)
    robot_radius = 20
    print("Initial image dimensions", width, height)
    print("scale", scale)
    print("Grid dimensions", grid_width, grid_height)
    print("Resized image dimensons", image.get_size())
    print(
        "Resized image dimension divided by scale", np.array(image.get_size()) * scale
    )
    m = Maze()
    m.generator = DungeonRooms(grid_height, grid_width)
    m.generate()
    grid = m.grid
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j] == 1:
                if i > 0 and grid[i - 1, j] == 1:
                    line_start = (j / scale / 2, i / scale / 2)
                    line_end = (j / scale / 2, (i - 1) / scale / 2)
                    draw_line(image, line_start, line_end)

                if j > 0 and grid[i, j - 1] == 1:
                    line_start = (j / scale / 2, i / scale / 2)
                    line_end = ((j - 1) / scale / 2, i / scale / 2)
                    draw_line(image, line_start, line_end)

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
