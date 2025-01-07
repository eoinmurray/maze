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
    """
    Generative model to predict observations given hidden states,
    plus a simple transition prior. We treat observation as walls vs. free cells,
    and define P(o|s). This keeps it as minimal as possible.
    """
    def __init__(self, state_dim=2, hidden_dim=16):
        super().__init__()
        # Observation likelihood
        self.obs_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # Transition prior (optional expansion if needed)
        self.trans_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state):
        # Predict observation
        obs_prob = self.obs_network(state)
        # Predict next state (not deeply used here, but placeholders for clarity)
        trans_pred = self.trans_network(state)
        return obs_prob, trans_pred

class ActiveInferenceAgent:
    """
    Agent that maintains beliefs over grid positions. We now define:
    1) Observation model: obs_prob = P(o|s)
    2) Prior preference: goal states have high preference
    3) Expected free energy: combines risk (deviation from preference),
       ambiguity (uncertainty about observations), and exploration.
    """
    def __init__(self, grid, start, goal):
        super().__init__()
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape
        
        # Prior preferences
        self.prior_preferences = self._create_prior_preferences()
        
        # Precision and exploration
        self.alpha = 1.0  
        self.beta = 0.5   
        
        # Initialize beliefs
        self.belief_states = self._initialize_beliefs()
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.belief_history = []
        self.full_path_history = []
        
        self.model = TinyGenerativeModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def _create_prior_preferences(self):
        prefs = np.zeros_like(self.grid, dtype=float)
        prefs[self.goal] = 1.0
        return prefs

    def _initialize_beliefs(self):
        beliefs = np.ones_like(self.grid, dtype=float) / (self.height * self.width)
        beliefs[self.start] = 0.1
        return beliefs / np.sum(beliefs)

    def compute_expected_free_energy(self, pos, next_pos):
        """
        1) Convert states to tensor.
        2) Get observation likelihood from generative model.
        3) Ambiguity = negative log precision about obs_prob.
        4) Risk = how unlikely next_pos is to align with the prior preference.
        5) Weighted by alpha, beta, minus preference.
        """
        current_state = torch.FloatTensor([pos[0], pos[1]])
        next_state = torch.FloatTensor([next_pos[0], next_pos[1]])

        obs_prob, _ = self.model(next_state)
        obs_prob_val = obs_prob.item()
        
        # Ambiguity: if obs_prob ~ 0.5, we have max uncertainty
        # We'll treat it here as negative entropy
        # For simplicity, interpret obs_prob as "prob of free cell"
        # => Entropy = - (p log p + (1-p) log(1-p))
        p = obs_prob_val
        ambiguity = -(p * np.log(p + 1e-10) + (1 - p) * np.log((1 - p) + 1e-10))

        # Risk: if next_pos is a wall but our obs_prob is high => mismatch
        is_free = (self.grid[next_pos] == 0)
        if is_free:
            risk = -np.log(p + 1e-10)  
        else:
            risk = -np.log((1 - p) + 1e-10)
        
        preference = self.prior_preferences[next_pos]
        efe = self.alpha * risk + self.beta * ambiguity - preference
        return efe

    def update_beliefs(self, pos):
        """
        After moving, incorporate new observation into beliefs.
        We just boost the probability of the observed cell.
        """
        likelihood = np.ones_like(self.grid, dtype=float)
        likelihood[pos] = 2.0
        self.belief_states *= likelihood
        self.belief_states /= np.sum(self.belief_states)

    def find_path(self):
        """
        Classical BFS-like path search, but picks next moves by minimal G.
        """
        current = self.start
        path = [current]
        visited = {current}
        
        while current != self.goal:
            actions_efe = []
            
            for dx, dy in self.actions:
                next_pos = (current[0] + dx, current[1] + dy)
                
                if (0 <= next_pos[0] < self.height 
                    and 0 <= next_pos[1] < self.width 
                    and self.grid[next_pos] != 1 
                    and next_pos not in visited):
                    
                    efe = self.compute_expected_free_energy(current, next_pos)
                    actions_efe.append((efe, next_pos))
            
            if not actions_efe:
                if len(path) <= 1:
                    return []
                path.pop()
                current = path[-1]
            else:
                best_action = min(actions_efe, key=lambda x: x[0])
                current = best_action[1]
                path.append(current)
                visited.add(current)

                self.update_beliefs(current)
                self._update_model(current)
                
                self.full_path_history.append(current)
                self.belief_history.append(self.belief_states.copy())
        
        return path

    def _update_model(self, pos):
        """
        Given the observed cell is free or wall, update model params
        via variational free energy (here simplified to cross-entropy + KL).
        """
        self.optimizer.zero_grad()
        x = torch.FloatTensor([pos[0], pos[1]])
        obs_prob, _ = self.model(x)

        # True label: 1.0 if free cell, else 0.0
        true_val = torch.FloatTensor([1.0 if self.grid[pos] == 0 else 0.0])
        
        # -log p(true_obs)
        neg_loglike = - (true_val * torch.log(obs_prob + 1e-10)
                         + (1 - true_val) * torch.log(1 - obs_prob + 1e-10))
        
        # Minimal KL placeholder (not fully used), e.g. L2 on parameters
        kl_divergence = 0
        for param in self.model.parameters():
            kl_divergence += torch.sum(param**2) * 1e-5
        
        loss = neg_loglike + kl_divergence
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
