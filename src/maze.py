import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class PrimsMaze:
    def __init__(self, size: int, seed: int = None):
        self.width = size if size % 2 == 1 else size + 1
        self.height = size if size % 2 == 1 else size + 1
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.maze = np.ones((self.height, self.width))
        self.walls: List[Tuple[int, int, int, int]] = []
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def is_valid_cell(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def generate(self):
        start_y = random.randrange(1, self.height - 1, 2)
        start_x = random.randrange(1, self.width - 1, 2)
        self.maze[start_y, start_x] = 0

        for dy, dx in self.directions:
            wall_y, wall_x = start_y + dy, start_x + dx
            if self.is_valid_cell(wall_y, wall_x):
                self.walls.append((wall_y, wall_x, start_y, start_x))

        while self.walls:
            wall_index = random.randrange(len(self.walls))
            wall_y, wall_x, from_y, from_x = self.walls.pop(wall_index)

            to_y = wall_y + (wall_y - from_y)
            to_x = wall_x + (wall_x - from_x)

            if self.is_valid_cell(to_y, to_x) and self.maze[to_y, to_x] == 1:
                self.maze[wall_y, wall_x] = 0
                self.maze[to_y, to_x] = 0

                for dy, dx in self.directions:
                    new_wall_y = to_y + dy
                    new_wall_x = to_x + dx
                    if self.is_valid_cell(new_wall_y, new_wall_x) and self.maze[new_wall_y, new_wall_x] == 1:
                        self.walls.append((new_wall_y, new_wall_x, to_y, to_x))

        return self.maze

    def visualize(self, save_path: str = None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        plt.grid(False)
        plt.axis('off')
        plt.title(f"Prim's Maze ({self.width}x{self.height})")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

class TerrainGenerator:
    def __init__(self, size: int, seed: int = None, obstacle_density: float = 0.2):
        self.width = size
        self.height = size
        self.seed = seed
        self.obstacle_density = obstacle_density
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.grid = np.zeros((self.height, self.width))

    def generate(self):
        size = self.width
        self.grid = np.zeros((size, size), dtype=int)
        
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        
        
        size = max(self.width, self.height)
        num_obstacles = int(size * size * self.obstacle_density)
        obstacle_positions = set()
        while len(obstacle_positions) < num_obstacles:
            pos = (random.randint(0, size-2), random.randint(0, size-2))
            if pos != (1, 1) and pos != (size-2, size-2):
                obstacle_positions.add(pos)
        for pos in obstacle_positions:
            self.grid[pos] = 1  # 1 represents an obstacle
          
        self.grid[(1, 1)] = 0
        self.grid[(size-2, size-2)] = 0
        return self.grid

    def visualize(self, save_path: str = None):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.grid, cmap='binary')
        plt.colorbar(label='Elevation')
        plt.grid(False)
        plt.axis('off')
        plt.title("Terrain")

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

def main():
    # Generate Prim's Maze
    print("Generating Prim's Maze")
    maze_gen = PrimsMaze(15, seed=42)
    maze = maze_gen.generate()
    maze_gen.visualize()

    # Generate Terrain
    print("Generating Terrain")
    terrain_gen = TerrainGenerator(15, seed=42, obstacle_density=0.3)
    terrain = terrain_gen.generate()
    terrain_gen.visualize()

if __name__ == "__main__":
    main()
