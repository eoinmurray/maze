import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from maze import TerrainGenerator, PrimsMaze

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyGenerativeModel(nn.Module):
    """
    Predict if a cell is free or a wall, plus a small transition prior.
    We'll still do a partial Bayesian update outside this network.
    """
    def __init__(self, state_dim=2, hidden_dim=32):
        super().__init__()
        self.obs_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.trans_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state):
        # state is [x, y]
        obs_prob = self.obs_network(state)
        trans_pred = self.trans_network(state)
        return obs_prob, trans_pred

class ActiveInferenceAgent:
    """
    A more canonical approach using Bayesian updates for beliefs
    and single-step expected free energy for action selection.
    """
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.height, self.width = grid.shape

        # Canonical hyperparameters
        self.alpha = 1.0  # weighting for risk
        self.beta = 1.0   # weighting for ambiguity
        self.gamma = 10.0 # controls action-selection sharpness
        self.goal_preference = 100.0
        self.visit_penalty = 1.0

        # Create prior preferences: closer to goal = higher reward
        self.prior_preferences = self._create_prior_preferences()

        # Beliefs: each cell is a potential "hidden state"
        self.beliefs = np.ones_like(grid, dtype=float)
        self.beliefs[self.start] *= 5.0  # we believe we're at the start
        self.beliefs[self.goal] *= 5.0
        self.beliefs /= np.sum(self.beliefs)

        # Keep track of path, visits, etc.
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.belief_history = []
        self.full_path_history = []
        self.visit_counts = np.zeros_like(grid, dtype=float)

        # PyTorch model for local observations
        self.model = TinyGenerativeModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-2)

    def _create_prior_preferences(self):
        prefs = np.zeros_like(self.grid, dtype=float)
        gx, gy = self.goal
        for x in range(self.height):
            for y in range(self.width):
                dist = math.sqrt((x - gx)**2 + (y - gy)**2)
                prefs[x, y] = self.goal_preference / (dist + 1)
        prefs[self.goal] = self.goal_preference
        return prefs

    def _bayesian_update(self, pos):
        """
        Update belief distribution q(s_t) given that we observed 'pos' is free or not.
        We'll use the model's predicted obs_prob to refine the entire grid's beliefs.
        This is a simplistic "global" update to illustrate the concept.
        """
        # 1) Predict obs_prob for each cell as if we were standing in that cell
        new_beliefs = np.zeros_like(self.beliefs)
        all_coords = [(x, y) for x in range(self.height) for y in range(self.width)]
        with torch.no_grad():
            for (x, y) in all_coords:
                state = torch.tensor([x, y], dtype=torch.float32, device=device)
                obs_prob, _ = self.model(state)
                # If the actual observation is "pos is free" => 1 if grid[pos]==0
                # We'll do a "likelihood" step for each cell
                # though logically you'd only do it for the cell you're in,
                # we do a global approximation for demonstration
                is_free = (self.grid[pos] == 0)
                p_obs_given_s = obs_prob.item() if is_free else (1 - obs_prob.item())
                # Posterior ~ prior * likelihood
                new_beliefs[x, y] = self.beliefs[x, y] * (p_obs_given_s + 1e-6)
        new_beliefs /= (np.sum(new_beliefs) + 1e-12)
        self.beliefs = new_beliefs

    def _transition_update(self):
        """
        This would normally incorporate p(s_{t+1} | s_t).
        For simplicity, we just diffuse beliefs across free cells.
        """
        new_beliefs = np.zeros_like(self.beliefs)
        for x in range(self.height):
            for y in range(self.width):
                if self.beliefs[x, y] > 0:
                    # spread that belief to neighbors
                    nbrs = []
                    for dx, dy in self.actions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx, ny] == 0:
                            nbrs.append((nx, ny))
                    if len(nbrs) == 0:
                        new_beliefs[x, y] += self.beliefs[x, y]
                    else:
                        for (nx, ny) in nbrs:
                            new_beliefs[nx, ny] += self.beliefs[x, y] / len(nbrs)
        new_beliefs /= (np.sum(new_beliefs) + 1e-12)
        self.beliefs = new_beliefs

    def _expected_free_energy(self, current, next_pos):
        """
        We do a single-step lookahead: what if we move to next_pos?
        risk = negative log probability of next_pos being free
        ambiguity = negative expected information gain
        preference = prior_preferences
        """
        # risk
        next_x, next_y = next_pos
        with torch.no_grad():
            st = torch.tensor([next_x, next_y], dtype=torch.float32, device=device)
            obs_prob, _ = self.model(st)
            obs_free_prob = obs_prob.item()

        # Risk: if free, good, if not free, big penalty
        is_free = (self.grid[next_pos] == 0)
        risk = -math.log(obs_free_prob + 1e-12) if is_free else -math.log((1 - obs_free_prob) + 1e-12)

        # Ambiguity: measure how uncertain our predicted observation is
        ambiguity = -(obs_free_prob * math.log(obs_free_prob + 1e-12) +
                      (1 - obs_free_prob) * math.log((1 - obs_free_prob) + 1e-12))

        # Preference from prior (we want to be near the goal)
        preference = self.prior_preferences[next_pos]

        # Visit penalty to avoid thrashing
        repeat_penalty = self.visit_penalty * self.visit_counts[next_pos]

        return self.alpha * risk + self.beta * ambiguity + repeat_penalty - preference

    def _choose_action(self, current):
        """
        Use a softmax over negative EFE (like a Boltzmann policy).
        """
        possible_actions = []
        for dx, dy in self.actions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < self.height and 0 <= ny < self.width and self.grid[nx, ny] != 1:
                efe = self._expected_free_energy(current, (nx, ny))
                possible_actions.append((efe, (nx, ny)))

        if not possible_actions:
            return current

        efe_values = [val for (val, _) in possible_actions]
        m = min(efe_values)
        shifted = [v - m for v in efe_values]
        expvals = [math.exp(-self.gamma * s) for s in shifted]
        total = sum(expvals)
        probs = [v / total for v in expvals]

        r = random.random()
        cumsum = 0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                return possible_actions[i][1]
        return possible_actions[-1][1]  # fallback

    def find_path(self, max_steps=500):
        current = self.start
        path = [current]
        steps = 0

        while current != self.goal and steps < max_steps:
            # Bayesian update from observation at current
            self._bayesian_update(current)

            # Basic transition update for next time
            self._transition_update()

            # Decide next action
            next_pos = self._choose_action(current)
            path.append(next_pos)
            self.full_path_history.append(next_pos)
            self.belief_history.append(self.beliefs.copy())

            # Count visits, move on
            self.visit_counts[next_pos] += 1.0
            self.visit_counts *= 0.99
            current = next_pos
            steps += 1

            # Update model’s parameters with current observation
            self._update_model(next_pos)

        return np.array(path)

    def _update_model(self, pos):
        """
        Train the generative model: if pos is free => label=1, else 0
        """
        self.optimizer.zero_grad()
        x, y = pos
        st = torch.tensor([x, y], dtype=torch.float32, device=device)
        obs_prob, _ = self.model(st)
        label = torch.tensor([1.0 if self.grid[pos] == 0 else 0.0], dtype=torch.float32, device=device)
        # negative log likelihood + small L2 regularization
        neg_ll = - (label * torch.log(obs_prob + 1e-12) + (1 - label) * torch.log(1 - obs_prob + 1e-12))
        reg = 0.0
        for p in self.model.parameters():
            reg += torch.sum(p**2) * 1e-5
        loss = neg_ll + reg
        loss.backward()
        self.optimizer.step()

class Visualizer:
    def __init__(self, agent: ActiveInferenceAgent):
        self.agent = agent
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.path_artists = []
        self.setup_plots()

    def setup_plots(self):
        self.fig.suptitle('Active Inference Pathfinding (More Canonical)')
        self.ax1.set_title('Environment & Path')
        self.grid_im = self.ax1.imshow(self.agent.grid, cmap='gray_r')
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.ax2.set_title('Belief States')
        self.belief_im = self.ax2.imshow(self.agent.beliefs, cmap='viridis', vmin=0, vmax=1)
        self.fig.colorbar(self.belief_im, ax=self.ax2)

    def update_frame(self, frame_num):
        while self.path_artists:
            artist = self.path_artists.pop()
            artist.remove()

        # Update belief
        if frame_num < len(self.agent.belief_history):
            data = self.agent.belief_history[frame_num]
            self.belief_im.set_array(data)
            self.belief_im.set_clim(vmin=data.min(), vmax=data.max())

        # Draw path
        if frame_num > 0 and frame_num <= len(self.agent.full_path_history):
            full_path = self.agent.full_path_history[:frame_num+1]
            path_x, path_y = zip(*full_path)
            for ax in [self.ax1, self.ax2]:
                line_plot, = ax.plot(path_y, path_x, 'r.-', linewidth=1, markersize=5)
                self.path_artists.append(line_plot)
                point, = ax.plot(path_y[-1], path_x[-1], 'go', markersize=12)
                self.path_artists.append(point)
                start, = ax.plot(self.agent.start[1], self.agent.start[0], 'b*', markersize=12)
                goal, = ax.plot(self.agent.goal[1], self.agent.goal[0], 'r*', markersize=12)
                self.path_artists.extend([start, goal])

        return [self.grid_im, self.belief_im] + self.path_artists

    def animate_path(self):
        anim = FuncAnimation(
            self.fig,
            self.update_frame,
            frames=len(self.agent.full_path_history) + 1,
            interval=100,
            blit=True,
            repeat=False
        )
        plt.show()

def main():
    size = 21
    random_integer = random.randint(0, 10000)
    print(f"Random integer: {random_integer}")

    # Maze generation
    grid = np.zeros((size, size), dtype=int)
    # maze_gen = PrimsMaze(size, seed=random_integer)
    maze_gen = TerrainGenerator(size, seed=random_integer, obstacle_density=0.3)
    grid = maze_gen.generate()

    agent = ActiveInferenceAgent(grid, start=(1, 1), goal=(size-2, size-2))
    path = agent.find_path(max_steps=500)
    print(path.shape)

    viz = Visualizer(agent)
    viz.animate_path()

if __name__ == "__main__":
    main()
