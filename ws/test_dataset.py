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


def circle_aa_rect_collision(center, radius, rect):
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


def draw_rects(surface, rects):
    for rect in rects:
        pygame.draw.rect(surface, (0, 255, 0), rect)


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
    maze_rects = []
    bbox_rects = []
    player_radius = 10
    player_pos = (
        random.randint(0 + player_radius, width - player_radius),
        random.randint(0 + player_radius, height - player_radius),
    )
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
        maze_rect = pygame.Rect(x_start, y_start, rect_width, rect_height)
        pygame.draw.rect(image, (0, 255, 0), maze_rect)
        maze_rects.append(maze_rect)
    for bbox in bboxes:
        left = bbox[0]
        top = bbox[1]
        width = bbox[2]
        height = bbox[3]
        bbox_rect = pygame.Rect(left, top, width, height)
        bbox_rects.append(bbox_rect)
    draw_rects(image, maze_rects)
    draw_rects(image, bbox_rects)
    pygame.draw.circle(image, (0, 0, 255), player_pos, player_radius)
    for rect in maze_rects:
        collided, coll_x, coll_y = circle_aa_rect_collision(
            player_pos, player_radius, rect
        )
        if collided:
            pygame.draw.circle(image, (255, 0, 0), (coll_x, coll_y), 2)
            print("Collision detected")
    for rect in bbox_rects:
        collided, coll_x, coll_y = circle_aa_rect_collision(
            player_pos, player_radius, rect
        )
        if collided:
            pygame.draw.circle(image, (255, 0, 0), (coll_x, coll_y), 2)
            print("Collision detected")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.blit(image, (0, 0))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
