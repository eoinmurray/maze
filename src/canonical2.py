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
        super().__init__()
        # Separate networks for mean and precision
        self.mean_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.precision_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

    def forward(self, x):
        mean = torch.sigmoid(self.mean_network(x))
        precision = self.precision_network(x)
        return mean, precision

class ActiveInferenceAgent:
    def __init__(self, grid, start, goal):
        super().__init__()
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape
        
        # Prior preferences (goal-directed behavior)
        self.prior_preferences = self._create_prior_preferences()
        
        # Expected free energy components
        self.alpha = 1.0  # precision parameter
        self.beta = 0.5   # exploration parameter
        
        self.belief_states = self._initialize_beliefs()
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.belief_history = []
        self.full_path_history = []
        
        self.model = TinyGenerativeModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def _create_prior_preferences(self):
        preferences = np.zeros_like(self.grid, dtype=float)
        preferences[self.goal] = 1.0
        return preferences

    def _initialize_beliefs(self):
        beliefs = np.ones_like(self.grid, dtype=float) / (self.height * self.width)
        beliefs[self.start] = 0.1
        return beliefs / np.sum(beliefs)

    def compute_expected_free_energy(self, pos, next_pos):
        # Convert positions to tensors
        current_state = torch.FloatTensor([pos[0], pos[1]])
        next_state = torch.FloatTensor([next_pos[0], next_pos[1]])
        
        # Get model predictions
        pred_mean, pred_precision = self.model(next_state)
        
        # Ambiguity term (epistemic value)
        ambiguity = -pred_precision.item()
        
        # Risk term (pragmatic value)
        risk = -torch.log(pred_mean + 1e-10).item() if self.grid[next_pos] == 0 else -torch.log(1 - pred_mean + 1e-10).item()
        
        # Prior preference term
        preference = self.prior_preferences[next_pos]
        
        # Expected free energy combines risk, ambiguity and preference
        return (self.alpha * risk + 
                self.beta * ambiguity - 
                preference)

    def update_beliefs(self, pos):
        # Bayesian belief update
        likelihood = np.ones_like(self.grid, dtype=float)
        likelihood[pos] = 2.0
        
        self.belief_states *= likelihood
        self.belief_states = self.belief_states / np.sum(self.belief_states)

    def find_path(self):
        current = self.start
        path = [current]
        visited = {current}
        
        while current != self.goal:
            actions_efe = []
            
            # Evaluate expected free energy for each action
            for dx, dy in self.actions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_pos[0] < self.height and 
                    0 <= next_pos[1] < self.width and 
                    self.grid[next_pos] != 1 and 
                    next_pos not in visited):
                    
                    efe = self.compute_expected_free_energy(current, next_pos)
                    actions_efe.append((efe, next_pos))
            
            if not actions_efe:
                if len(path) <= 1:
                    return []
                path.pop()
                current = path[-1]
            else:
                # Select action with minimal expected free energy
                best_action = min(actions_efe, key=lambda x: x[0])
                current = best_action[1]
                path.append(current)
                visited.add(current)
                
                # Update beliefs and model
                self.update_beliefs(current)
                self._update_model(current)
                
                self.full_path_history.append(current)
                self.belief_history.append(self.belief_states.copy())
        
        return path

    def _update_model(self, pos):
        self.optimizer.zero_grad()
        x = torch.FloatTensor([pos[0], pos[1]])
        mean, precision = self.model(x)
        
        # Compute variational free energy
        true_value = torch.FloatTensor([0.0 if self.grid[pos] == 0 else 1.0])
        neg_log_likelihood = -torch.log(mean + 1e-10) if true_value.item() == 0 else -torch.log(1 - mean + 1e-10)
        kl_divergence = 0.5 * (precision + torch.log(1/precision) - 1)
        
        loss = neg_log_likelihood + kl_divergence
        loss.backward()
        self.optimizer.step()

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
