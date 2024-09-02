from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

from mapper import map_points
from settings import *
from picker import pick_points
from tracker import track_features
from utils import filter_outliers

class Plotter:

    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.prev_frame = None
        self.points = None

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-1, 1)

        self.scat = ax.scatter([], [], [], c=[])

        self.animation = FuncAnimation(fig, self.update_plot, frames=frame_count, interval=500, blit=False)

        plt.show()

    def update_plot(self, frame):

        ret, frame = self.video_capture.read()

        assert(ret)
        
        if self.prev_frame is None:
            self.points = pick_points(frame)
            self.prev_frame = frame
            return
        
        self.points = track_features(self.prev_frame, frame, self.points)

        map_points(self.points)

        points_filtered = filter_outliers(self.points, z_threshold=3)

        real_positions = np.array([point.real_position for point in points_filtered])

        real_positions /= np.max(real_positions, axis=0)

        colors = np.array([point.color for point in points_filtered]) / 255

        self.scat._offsets3d = ((real_positions[:, 0], real_positions[:, 2], -real_positions[:, 1]))
        self.scat.set_facecolors(colors)
        self.scat.set_edgecolors(colors)

        self.prev_frame = frame

        return self.scat