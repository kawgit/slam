import cv2
import numpy as np

from settings import *

def filter_outliers(points, z_threshold=3):
    position_3ds = np.array([point.position_3d for point in points])

    mean = np.mean(position_3ds, axis=0)
    std_dev = np.std(position_3ds, axis=0)
    z_scores = np.abs((position_3ds - mean) / std_dev)
    
    return [point for point, z_score in zip(points, z_scores) if all(z_score < z_threshold)]

def update_weighted_time_series(old, new, new_weight):

    if old is None:
        return new
    
    old_weight = 1 - new_weight

    return new_weight * new + old_weight * old