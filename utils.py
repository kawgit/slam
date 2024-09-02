import cv2
import numpy as np

from settings import *

def filter_outliers(points, z_threshold=3):
    real_positions = np.array([point.real_position for point in points])

    mean = np.mean(real_positions, axis=0)
    std_dev = np.std(real_positions, axis=0)
    z_scores = np.abs((real_positions - mean) / std_dev)
    
    return [point for point, z_score in zip(points, z_scores) if all(z_score < z_threshold)]
