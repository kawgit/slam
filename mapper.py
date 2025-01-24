import numpy as np

from settings import *

def map_point(frame_point):

    frame_x, frame_y = frame_point.position_2d - np.array([frame_width, frame_height]) / 2

    frame_speed = np.linalg.norm(frame_point.velocity_2d)

    real_z = focal_length / (frame_speed + .0000001)
    real_x = real_z * frame_x / focal_length
    real_y = real_z * frame_y / focal_length

    frame_point.update_3d(np.array([real_x, real_y, real_z]))

def map_points(frame_points):

    for frame_point in frame_points:
        map_point(frame_point)