import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
from maze import PrimsMaze

class ActiveInferenceAgent:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape
        
        # Initialize beliefs and parameters
        self.belief_states = np.zeros_like(grid)
        self.belief_states[start[0], start[1]] = 1.0
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
        self.temperature = 1.0  # exploration parameter
        self.belief_history = []
        self.full_path_history = [] 
        
    def update_beliefs(self, pos: Tuple[int, int]):
        """Update belief states based on current position."""
        self.belief_states *= 0.95
        self.belief_states[pos[0], pos[1]] += 0.05
        self.belief_states /= np.sum(self.belief_states)
        self.belief_history.append(self.belief_states.copy())
        
    def find_path(self) -> List[Tuple[int, int]]:
        """Find path using active inference."""
        current = self.start
        path = [current]
        visited = {current}
        
        self.full_path_history.append(current)
        
        while current != self.goal:
            best_energy = float('inf')
            best_next = None
            
            # Evaluate possible actions
            for dx, dy in self.actions:
                x, y = current[0] + dx, current[1] + dy
                if (0 <= x < self.height and 0 <= y < self.width and 
                    self.grid[x, y] != 1 and (x, y) not in visited):
                    
                    # Compute free energy (distance to goal + terrain cost)
                    energy = (abs(x - self.goal[0]) + abs(y - self.goal[1]) + 
                            self.grid[x, y] - self.temperature * 
                            np.log(self.belief_states[x, y] + 1e-10))
                    
                    if energy < best_energy:
                        best_energy = energy
                        best_next = (x, y)
            
            if best_next is None:
                if len(path) <= 1:
                    return []
                path.pop()
                current = path[-1]
            else:
                current = best_next
                path.append(current)
                visited.add(current)
                self.update_beliefs(current)
                self.temperature *= 0.995
                
            self.full_path_history.append(current)

        return path

class Visualizer:
    def __init__(self, agent: ActiveInferenceAgent):
        self.agent = agent
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.setup_plots()
        self.path_artists = []
        
    def setup_plots(self):
        """Initialize the visualization plots."""
        self.fig.suptitle('Active Inference Pathfinding')
        
        # Environment plot
        self.ax1.set_title('Environment & Path')
        self.grid_im = self.ax1.imshow(self.agent.grid, cmap='gray_r')
        
        # Belief states plot
        self.ax2.set_title('Belief States')
        self.belief_im = self.ax2.imshow(self.agent.belief_states, cmap='viridis')
        self.fig.colorbar(self.belief_im, ax=self.ax2)
        
    def update_frame(self, frame_num):
        """Update animation frame."""
        # Remove previous frame's artists
        while self.path_artists:
            artist = self.path_artists.pop()
            artist.remove()

        # Update belief map with the belief state at the current frame
        if frame_num < len(self.agent.belief_history):
            self.belief_im.set_array(self.agent.belief_history[frame_num])

        # Plot full path up to the current frame
        if frame_num > 0:
            full_path = self.agent.full_path_history[:frame_num+1]
            path_x, path_y = zip(*full_path)

            for ax in [self.ax1, self.ax2]:
                # Plot the full path (including backtracking)
                full_line, = ax.plot(path_y, path_x, 'r.-', linewidth=1, markersize=5)
                self.path_artists.append(full_line)

                # Plot current position
                point, = ax.plot(path_y[-1], path_x[-1], 'go', markersize=12)
                self.path_artists.append(point)

                # Plot start and goal
                start, = ax.plot(self.agent.start[1], self.agent.start[0], 'b*', markersize=12)
                goal, = ax.plot(self.agent.goal[1], self.agent.goal[0], 'r*', markersize=12)
                self.path_artists.extend([start, goal])

        return [self.grid_im, self.belief_im] + self.path_artists
    
    def animate_path(self):
        """Animate the pathfinding process."""
        anim = FuncAnimation(self.fig, self.update_frame, frames=len(self.agent.full_path_history),
                           interval=500, blit=True, repeat=False)
        plt.show()

def main():
    maze_gen = PrimsMaze(9, 9, seed=42)
    grid = maze_gen.generate()
    agent = ActiveInferenceAgent(grid, start=(1, 1), goal=(7, 7))
    path = agent.find_path()
    
    viz = Visualizer(agent)
    viz.animate_path()

if __name__ == "__main__":
    main()