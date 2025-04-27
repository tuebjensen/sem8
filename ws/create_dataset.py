import json
import math
import os

import pygame
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms

from maze_generator import (
    lines_from_maze,
    rooms_from_bboxes,
    scale_bboxes,
)


def images_iterator(images, annotations):
    for image_object in images:
        image_file_name = image_object["file_name"]
        image_dir = "images"
        image_path = os.path.join(image_dir, image_file_name)
        image = pygame.image.load(image_path)
        image_id = image_object["id"]
        bboxes = []
        category_ids = []
        for annotation in annotations:
            if annotation["image_id"] == image_id:
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                category_ids.append(category_id)
                bboxes.append(bbox)
        yield image, bboxes, category_ids


def main():
    pygame.init()
    desired_grid_width = 5
    desired_grid_height = 10
    # Maze density
    desired_grid_area = desired_grid_width * desired_grid_height
    m = Maze()
    annotation_file = "images/instances_default.json"
    with open(annotation_file) as f:
        annotation_dict = json.load(f)
    images = annotation_dict["images"]
    categories = annotation_dict["categories"]
    annotations = annotation_dict["annotations"]
    out_folder = "output"
    os.makedirs(out_folder, exist_ok=True)
    meta_data = {"categories": categories, "images": []}
    for i, (image, bboxes, category_ids) in enumerate(
        images_iterator(images, annotations)
    ):
        print(f"Processing image {i + 1}")
        width, height = image.get_size()
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
        rooms = rooms_from_bboxes(bboxes, scale)
        mazes = []
        m.generator = DungeonRooms(grid_height, grid_width, rooms=rooms)
        for _ in range(5):
            m.generate()
            maze = m.grid
            lines = lines_from_maze(maze, scale)
            mazes.append(lines)
        bbox_objects = []
        for bbox, category_id in zip(bboxes, category_ids):
            bbox_object = {
                "bbox": bbox,
                "category_id": category_id,
            }
            bbox_objects.append(bbox_object)
        image_file_name = f"{i}.jpg"
        image_annotation = {
            "file_name": image_file_name,
            "bbox_objects": bbox_objects,
            "mazes": mazes,
        }
        meta_data["images"].append(image_annotation)
        # image = draw_bboxes(image, bboxes)
        # draw_maze(image, maze, scale)

        file_path = os.path.join(out_folder, image_file_name)
        pygame.image.save(image, file_path)
    pygame.quit()

    file_path = os.path.join(out_folder, "meta_data.json")
    with open(file_path, "w") as f:
        json.dump(meta_data, f, indent=4)


if __name__ == "__main__":
    main()
