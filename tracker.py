import numpy as np
import cv2

from settings import *

def track_features(prev_frame, curr_frame, points):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    points_np = np.array([point.position_2d.reshape(1, 2) for point in points])

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    points_np_new, status, error = cv2.calcOpticalFlowPyrLK(prev_frame_gray, curr_frame_gray, points_np, None, **lk_params)
    
    for point, point_np_new in zip(points, points_np_new.reshape(-1, 2)):
        point.update_2d(point_np_new)

    return [point for point, status in zip(points, status.reshape(-1)) if status == 1]