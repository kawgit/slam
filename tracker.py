import numpy as np
import cv2

from settings import *

def track_features(prev_frame, curr_frame, prev_points):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_points_cv2 = np.array([prev_point.position.reshape(1, 2) for prev_point in prev_points])

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    curr_points_cv2, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, prev_points_cv2, None, **lk_params)
    
    curr_points = [prev_point.update(curr_point_cv2) for prev_point, curr_point_cv2 in zip(prev_points, curr_points_cv2.reshape(-1, 2))]
    
    return [point for point, status in zip(curr_points, status.reshape(-1)) if status == 1]