from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import cv2

from mapper import map_points
from settings import *
from picker import pick_points
from tracker import track_features
from utils import filter_outliers

class Plotter:

    def __init__(self, video_capture):
        self.video_capture = video_capture
        self.prev_frame = None
        self.points = []

        self.fig = plt.figure(figsize=(15, 5))

        # 3D plot
        self.ax_3d = self.fig.add_subplot(131, projection='3d')
        self.ax_3d.set_proj_type('persp')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Z')
        self.ax_3d.set_zlabel('Y')
        self.ax_3d.set_xlim(-3, 3)
        self.ax_3d.set_ylim(0, 3)
        self.ax_3d.set_zlim(-3, 3)
        self.scat = self.ax_3d.scatter([], [], [], c=[])

        # Grayscale distance map
        self.ax_image = self.fig.add_subplot(132)
        self.ax_image.axis('off')
        self.image_plot = self.ax_image.imshow(
            np.zeros((frame_height, frame_width)), 
            cmap='gray',
            vmin=0,
            vmax=255
        )

        # Unaltered frame in color
        self.ax_color_image = self.fig.add_subplot(133)
        self.ax_color_image.axis('off')
        self.color_image_plot = self.ax_color_image.imshow(
            np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        )

        self.animation = FuncAnimation(self.fig, self.update_plot, frames=frame_count, interval=500, blit=False)

        plt.show()

    def update_plot(self, frame):

        ret, frame = self.video_capture.read()
        assert ret

        # Display unaltered color frame
        self.color_image_plot.set_data(frame[..., ::-1])

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.points = pick_points(frame)
            self.prev_frame = frame
            self.image_plot.set_data(gray_frame)
            return

        self.points = track_features(self.prev_frame, frame, self.points)
        map_points(self.points)
        points_filtered = filter_outliers(self.points, z_threshold=3)

        position_3ds = np.array([point.position_3d for point in points_filtered])
        position_3ds /= np.median(np.abs(position_3ds), axis=0) * 2
        colors = np.array([point.color for point in points_filtered]) / 255

        self.scat._offsets3d = (position_3ds[:, 0], position_3ds[:, 2], -position_3ds[:, 1])
        self.scat.set_facecolors(colors)
        self.scat.set_edgecolors(colors)

        distances = np.linalg.norm(position_3ds, axis=1)
        distance_map = np.ones_like(gray_frame, dtype=np.float32) * 10

        sorted_points_and_distances = sorted(list(zip(points_filtered, distances)), key=lambda x: x[1])
        for point, distance in sorted_points_and_distances:
            x, y = int(point.position_2d[0]), int(point.position_2d[1])
            if 0 <= x < gray_frame.shape[1] and 0 <= y < gray_frame.shape[0]:
                cv2.circle(distance_map, (x, y), radius=10, color=distance, thickness=-1)

        distance_map = np.log(distance_map)
        distance_map -= distance_map.min()
        distance_map /= distance_map.max()
        distance_map *= 255

        self.image_plot.set_data(distance_map)

        self.prev_frame = frame

        return self.scat, self.image_plot, self.color_image_plot
