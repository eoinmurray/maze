import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple

class PrimsMaze:
    def __init__(self, width: int, height: int, seed: int = None):
        """
        Initialize maze generator.
        width and height should be odd numbers to ensure proper wall structure.
        """
        # Ensure odd dimensions for proper wall placement
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.maze = np.ones((self.height, self.width))
        self.walls: List[Tuple[int, int, int, int]] = []
        
        # Directions for moving: right, down, left, up
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
    def is_valid_cell(self, y: int, x: int) -> bool:
        """Check if cell coordinates are within maze bounds."""
        return 0 <= y < self.height and 0 <= x < self.width
        
    def generate(self):
        """Generate maze using Prim's algorithm."""
        # Start from a random odd-numbered cell
        start_y = random.randrange(1, self.height - 1, 2)
        start_x = random.randrange(1, self.width - 1, 2)
        self.maze[start_y, start_x] = 0
        
        # Add surrounding walls to list
        for dy, dx in self.directions:
            wall_y, wall_x = start_y + dy, start_x + dx
            if self.is_valid_cell(wall_y, wall_x):
                self.walls.append((wall_y, wall_x, start_y, start_x))
        
        # Process walls
        while self.walls:
            # Pick random wall
            wall_index = random.randrange(len(self.walls))
            wall_y, wall_x, from_y, from_x = self.walls.pop(wall_index)
            
            # Calculate the cell on opposite side of wall
            to_y = wall_y + (wall_y - from_y)
            to_x = wall_x + (wall_x - from_x)
            
            # If opposite cell is valid and unvisited, carve passage
            if (self.is_valid_cell(to_y, to_x) and 
                self.maze[to_y, to_x] == 1):
                # Carve passage
                self.maze[wall_y, wall_x] = 0  # Remove wall
                self.maze[to_y, to_x] = 0      # Carve destination
                
                # Add new walls
                for dy, dx in self.directions:
                    new_wall_y = to_y + dy
                    new_wall_x = to_x + dx
                    if (self.is_valid_cell(new_wall_y, new_wall_x) and 
                        self.maze[new_wall_y, new_wall_x] == 1):
                        self.walls.append((new_wall_y, new_wall_x, to_y, to_x))
        
        return self.maze
    
    def visualize(self, save_path: str = None):
        """Visualize the maze."""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        plt.grid(False)
        plt.axis('off')
        plt.title(f"Prim's Maze ({self.width}x{self.height})")
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

def main():
    # Generate mazes of different sizes
    
    # Small maze (15x15)
    maze_gen = PrimsMaze(9, 9, seed=42)
    maze_small = maze_gen.generate()
    maze_gen.visualize()
    
    # Medium maze (25x25)
    maze_gen = PrimsMaze(25, 25, seed=42)
    maze_medium = maze_gen.generate()
    maze_gen.visualize()
    
    # Large maze (35x35)
    maze_gen = PrimsMaze(35, 35, seed=42)
    maze_large = maze_gen.generate()
    maze_gen.visualize()
    
    # Different seed
    maze_gen = PrimsMaze(25, 25, seed=123)
    maze_diff_seed = maze_gen.generate()
    maze_gen.visualize()

if __name__ == "__main__":
    main()