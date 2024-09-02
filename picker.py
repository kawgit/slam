import numpy as np
import cv2

from point import Point
from settings import *

def calculate_average_color(color_frame, x, y, region_size=5):
    x_start = max(x - region_size // 2, 0)
    y_start = max(y - region_size // 2, 0)
    x_end = min(x + region_size // 2 + 1, frame_width)
    y_end = min(y + region_size // 2 + 1, frame_height)

    region = color_frame[y_start:y_end, x_start:x_end]

    avg_color = np.mean(region, axis=(0, 1)).astype(int)

    assert(max(avg_color) <= 255)
    assert(min(avg_color) >= 0)

    return avg_color

def pick_points(frame):
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_colored = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    feature_params = dict(maxCorners=num_points_per_chunk,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    chunk_width = frame_width // num_chunks_x
    chunk_height = frame_height // num_chunks_y

    points = []
    for i in range(num_chunks_x):
        for j in range(num_chunks_y):

            chunk = frame_grayscale[j * chunk_height:(j + 1) * chunk_height, i * chunk_width:(i + 1) * chunk_width]
            chunk_points = cv2.goodFeaturesToTrack(chunk, **feature_params)

            if chunk_points is not None:

                chunk_points[:, 0, 0] += i * chunk_width
                chunk_points[:, 0, 1] += j * chunk_height

                points.extend(chunk_points)

    return [Point(point.reshape(-1), color=calculate_average_color(frame_colored, int(point[0, 0]), int(point[0, 1]), region_size=5)) for point in points]
