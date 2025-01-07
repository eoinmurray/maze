import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, agent):
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
