import json
import math
import os
import random
import sys
import timeit

import pygame
from mazelib.generate.DungeonRooms import DungeonRooms
from tqdm import tqdm


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


def rooms_from_bboxes(bbox_rects, cell_size):
    # Calculate the room coordinates in the grid
    # The coordinates are in the format (y, x) for the grid
    # Multiply by 2 for similar reasons as in draw_maze
    rooms = []
    for rect in bbox_rects:
        x, y, w, h = rect
        top_left_coord = (math.floor(y * cell_size * 2), math.floor(x * cell_size * 2))
        bottom_right_coord = (
            math.floor((y + h) * cell_size * 2),
            math.floor((x + w) * cell_size * 2),
        )
        room = [top_left_coord, bottom_right_coord]
        rooms.append(room)
    return rooms


def room_from_circle(center, radius, cell_size):
    x, y = center
    top_left_coord = (
        math.floor((y - radius) * cell_size * 2),
        math.floor((x - radius) * cell_size * 2),
    )
    bottom_right_coord = (
        math.floor((y + radius) * cell_size * 2),
        math.floor((x + radius) * cell_size * 2),
    )
    room = [top_left_coord, bottom_right_coord]
    return room


def scale_bboxes(bboxes, x_scale, y_scale):
    scaled_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        scaled_bboxes.append(
            pygame.Rect(
                (
                    round(x * x_scale),
                    round(y * y_scale),
                    round(w * x_scale),
                    round(h * y_scale),
                )
            )
        )
    return scaled_bboxes


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


# Pseudo code for the recursive function (note that this is not exactly how it is implemented right now)

# def find_acceptable_object_image():
#     pass


# def place_object(object_image):
#     pass


# def check_placement(top, left, object_image):
#     pass


# def pseudo_function(sampled_categories, max_tries):
#     if len(sampled_categories) == 0:
#         return []

#     category = sampled_categories[0]
#     object_image = find_acceptable_object_image()
#     for _ in range(max_tries):
#         top, left = place_object(object_image)
#         successful_placement = check_placement(top, left, object_image)
#         if successful_placement:
#             child_object_images = pseudo_function(sampled_categories[1:], max_tries)
#             if child_object_images is not None:
#                 return [(top, left, object_image)].extend(child_object_images)

#     return None


def generate_image():
    def find_and_place_objects(sampled_categories, max_tries: int, current_rects=[]):
        if len(sampled_categories) == 0:
            return []
        category = sampled_categories[0]
        category_dir = os.path.join(data_folder, category)
        category_files = os.listdir(category_dir)
        acceptable_image_found = False
        object_image = pygame.Surface((0, 0))
        while not acceptable_image_found:
            image_name = random.choice(category_files)
            image_path = os.path.join(category_dir, image_name)
            object_image = pygame.image.load(image_path)
            object_width, object_height = object_image.get_size()
            if object_width > min_object_size or object_height > min_object_size:
                acceptable_image_found = True

        object_width, object_height = object_image.get_size()
        if object_width > max_object_size and object_width >= object_height:
            new_object_width = random.randint(min_object_size, max_object_size)
            scale = max(new_object_width / object_width, min_object_size / object_width)
            object_image = pygame.transform.scale_by(object_image, scale)
        elif object_height > max_object_size and object_height >= object_width:
            new_object_height = random.randint(min_object_size, max_object_size)
            scale = max(
                new_object_height / object_height, min_object_size / object_height
            )
            object_image = pygame.transform.scale_by(object_image, scale)

        object_rect = object_image.get_rect()
        for _ in range(max_tries):
            object_rect.x = random.randint(0, width - object_rect.width)
            object_rect.y = random.randint(0, height - object_rect.height)
            # collidelist returns -1 if there is no collision
            overlapping = object_rect.collidelist(current_rects) != -1
            if not overlapping:
                child_object_images = find_and_place_objects(
                    sampled_categories[1:], max_tries, current_rects + [object_rect]
                )
                if child_object_images is not None:
                    images_and_rects = [
                        (object_rect, object_image)
                    ] + child_object_images
                    return images_and_rects
        return None

    data_folder = "data"
    categories = [
        f
        for f in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, f))
    ]
    width, height = 441, 441
    max_object_size = 125
    # The object size only has to satisfy one of the minimums so a 150x50 object is acceptable
    min_object_size = 80
    image = pygame.Surface((width, height))
    image.fill((100, 100, 100))
    n_categories = 7
    sampled_categories = random.sample(categories, n_categories)
    max_tries = 10

    object_images_and_rects = None
    while object_images_and_rects is None:
        object_images_and_rects = find_and_place_objects(sampled_categories, max_tries)

    object_rects = []
    for object_rect, object_image in object_images_and_rects:
        object_rects.append(object_rect)
        image.blit(object_image, object_rect)
    return image, object_rects, sampled_categories


def make_maze_rects_from_lines(maze_lines, image_width, image_height):
    maze_rects = []
    for line in maze_lines:
        line_width = 2
        x_start = math.floor(line["line_start"][0])
        y_start = math.floor(line["line_start"][1])
        x_end = math.floor(line["line_end"][0])
        y_end = math.floor(line["line_end"][1])
        rect_width = max(abs(x_end - x_start), line_width)
        rect_height = max(abs(y_end - y_start), line_width)
        x_start = min(x_start, image_width - line_width)
        y_start = min(y_start, image_height - line_width)
        maze_rect = pygame.Rect(x_start, y_start, rect_width, rect_height)
        maze_rects.append(maze_rect)
    return maze_rects


def generate_maze(
    maze_object,
    image,
    bbox_rects,
    spawn_point=None,
    agent_radius=None,
):
    desired_grid_width = 7
    desired_grid_height = 7
    # Maze density
    desired_grid_area = desired_grid_width * desired_grid_height
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
    bbox_rects = scale_bboxes(bbox_rects, bbox_x_scale, bbox_y_scale)
    rooms = []
    rooms.extend(rooms_from_bboxes(bbox_rects, scale))
    if spawn_point is not None and agent_radius is not None:
        rooms.append(room_from_circle(spawn_point, agent_radius, scale))
    maze_object.generator = DungeonRooms(grid_height, grid_width, rooms=rooms)
    maze_object.generate()
    maze = maze_object.grid
    maze_lines = lines_from_maze(maze, scale)
    maze_rects = make_maze_rects_from_lines(
        maze_lines, new_image_width, new_image_height
    )
    return image, maze_rects, bbox_rects


def main():
    import Sem8Env

    env = Sem8Env.Sem8Env()
    n_images = 10
    validation_data = []
    data_dir = "test"
    for i in tqdm(range(n_images)):
        env.reset()
        image_file_name = f"val_image_{i}.png"
        image_path = os.path.join(data_dir, image_file_name)
        os.makedirs(data_dir, exist_ok=True)
        image = env._image
        bbox_rects = env._bbox_rects
        bboxes = [(x, y, w, h) for x, y, w, h in bbox_rects]
        categories = env._categories
        target_bbox_index = env._target_bbox_index
        agent_position = env._agent_position.tolist()
        agent_angle = int(env._agent_angle)
        maze_rects = env._maze_rects
        maze_xywh = [(x, y, w, h) for x, y, w, h in maze_rects]
        prompt = env._prompt
        validation_data.append(
            {
                "image_file_name": image_file_name,
                "bboxes": bboxes,
                "categories": categories,
                "target_bbox_index": target_bbox_index,
                "agent_position": agent_position,
                "agent_angle": agent_angle,
                "maze_xywh": maze_xywh,
                "prompt": prompt,
            }
        )
        # Save the image
        pygame.image.save(image, image_path)

    # Save the validation data to a JSON file
    with open(os.path.join(data_dir, "meta_data.json"), "w") as f:
        json.dump(validation_data, f, indent=4)


if __name__ == "__main__":
    main()
