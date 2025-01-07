import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from maze import PrimsMaze
import random

class TinyGenerativeModel(nn.Module):
    def __init__(self, state_dim=2, hidden_dim=16):
        super(TinyGenerativeModel, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # predict probability of being free space vs. wall

    def forward(self, x):
        # Sigmoid output for "probability" of free space
        h = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(h))

class ActiveInferenceAgent:
    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape

        self.belief_states = np.zeros_like(grid, dtype=float)
        self.belief_states[start[0], start[1]] = 1.0
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.temperature = 1.0
        self.belief_history = []
        self.full_path_history = []

        # Tiny generative model
        self.model = TinyGenerativeModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def update_beliefs(self, pos: Tuple[int, int]):
        self.belief_states *= 0.95
        self.belief_states[pos[0], pos[1]] += 0.05
        self.belief_states /= np.sum(self.belief_states)

    def compute_free_energy(self, pos: Tuple[int, int]):
        # Convert current position to PyTorch tensor
        x_input = torch.FloatTensor([pos[0], pos[1]])
        pred_prob = self.model(x_input)
        # Negative log likelihood (treat 0=free, 1=wall), invert because we want free
        true_value = torch.FloatTensor([0.0 if self.grid[pos[0], pos[1]]==0 else 1.0])
        loss = -torch.log(pred_prob + 1e-10) if true_value.item() == 0 else -torch.log(1 - pred_prob + 1e-10)
        return loss

    def learn_generative_params(self, pos: Tuple[int, int]):
        self.optimizer.zero_grad()
        loss = self.compute_free_energy(pos)
        loss.backward()
        self.optimizer.step()

    def find_path(self) -> List[Tuple[int, int]]:
        current = self.start
        path = [current]
        visited = {current}
        
        self.full_path_history.append(current)
        self.belief_history.append(self.belief_states.copy())

        while current != self.goal:
            best_energy = float('inf')
            best_next = None

            for dx, dy in self.actions:
                x, y = current[0] + dx, current[1] + dy
                if (0 <= x < self.height and 0 <= y < self.width
                    and self.grid[x, y] != 1 and (x, y) not in visited):

                    # Incorporate distance cost, environment cost, and predicted free energy
                    dist = abs(x - self.goal[0]) + abs(y - self.goal[1])
                    cost = self.grid[x, y]
                    log_belief = np.log(self.belief_states[x, y] + 1e-10)

                    # Estimate "surprise" with the tiny generative model
                    with torch.no_grad():
                        free_energy = self.compute_free_energy((x, y)).item()

                    energy = dist + cost - self.temperature * log_belief + free_energy
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

                # Update model parameters
                self.learn_generative_params(current)

            self.full_path_history.append(current)
            self.belief_history.append(self.belief_states.copy())

        return path

class Visualizer:
    def __init__(self, agent: ActiveInferenceAgent):
        self.agent = agent
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.setup_plots()
        self.path_artists = []

    def setup_plots(self):
        self.fig.suptitle('Active Inference Pathfinding')
        self.ax1.set_title('Environment & Path')
        self.grid_im = self.ax1.imshow(self.agent.grid, cmap='gray_r')
        self.ax2.set_title('Belief States')
        self.belief_im = self.ax2.imshow(self.agent.belief_states, cmap='viridis')
        self.fig.colorbar(self.belief_im, ax=self.ax2)

    def update_frame(self, frame_num):
        while self.path_artists:
            artist = self.path_artists.pop()
            artist.remove()

        if frame_num < len(self.agent.belief_history):
            self.belief_im.set_array(self.agent.belief_history[frame_num])

        if frame_num > 0:
            full_path = self.agent.full_path_history[:frame_num+1]
            path_x, path_y = zip(*full_path)

            for ax in [self.ax1, self.ax2]:
                full_line, = ax.plot(path_y, path_x, 'r.-', linewidth=1, markersize=5)
                self.path_artists.append(full_line)

                point, = ax.plot(path_y[-1], path_x[-1], 'go', markersize=12)
                self.path_artists.append(point)

                start, = ax.plot(self.agent.start[1], self.agent.start[0], 'b*', markersize=12)
                goal, = ax.plot(self.agent.goal[1], self.agent.goal[0], 'r*', markersize=12)
                self.path_artists.extend([start, goal])

        return [self.grid_im, self.belief_im] + self.path_artists

    def animate_path(self):
        anim = FuncAnimation(self.fig,
                             self.update_frame,
                             frames=len(self.agent.full_path_history),
                             interval=50,
                             blit=True,
                             repeat=False)
        plt.show()

def main():
    size = 21
    
    random_integer = random.randint(0, 100)
    print(f"Random integer: {random_integer}")
    
    maze_gen = PrimsMaze(size, size, seed=random_integer)
    grid = maze_gen.generate()
    agent = ActiveInferenceAgent(grid, start=(1, 1), goal=(size-2, size-2))
    path = agent.find_path()

    viz = Visualizer(agent)
    viz.animate_path()

if __name__ == "__main__":
    main()
