import numpy as np
from mazelib import Maze
from mazelib.generate.DungeonRooms import DungeonRooms
import matplotlib.pyplot as plt


def showPNG(grid):
    """Generate a simple image of the maze."""
    plt.figure(figsize=(10, 5))
    plt.imshow(grid, cmap=plt.cm.binary, interpolation="nearest")
    plt.xticks([]), plt.yticks([])
    plt.show()


m = Maze()
m.generator = DungeonRooms(5, 5)
m.generate()
grid = m.grid
print(grid)
showPNG(grid)
