import json
import math
import os
import random

import pygame

from maze_generator import draw_bboxes, draw_lines


def load_random_image_bbox_maze(annotation_dict):
    image_object = random.choice(annotation_dict["images"])
    image_file_name = image_object["file_name"]
    image_dir = "output"
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


def main():
    pygame.init()
    annotations_path = os.path.join("output", "meta_data.json")
    with open(annotations_path) as f:
        annotation_dict = json.load(f)
    image, bboxes, category_ids, maze_lines = load_random_image_bbox_maze(
        annotation_dict
    )
    width, height = image.get_size()
    window = pygame.display.set_mode((width, height))
    running = True
    line_rects = []
    for line in maze_lines:
        line_width = 2
        x_start = math.floor(line["line_start"][0])
        y_start = math.floor(line["line_start"][1])
        x_end = math.floor(line["line_end"][0])
        y_end = math.floor(line["line_end"][1])
        rect_width = max(abs(x_end - x_start), line_width)
        rect_height = max(abs(y_end - y_start), line_width)
        x_start = min(x_start, width - line_width)
        y_start = min(y_start, height - line_width)
        line_rect = pygame.Rect(x_start, y_start, rect_width, rect_height)
        pygame.draw.rect(image, (0, 255, 0), line_rect)
        line_rects.append(line_rect)

    draw_bboxes(image, bboxes)
    # draw_lines(image, maze_lines)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.blit(image, (0, 0))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
