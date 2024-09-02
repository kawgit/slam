import cv2
import numpy as np

from mapper import map_points
from picker import pick_points
from plotter import draw_tracks, plot_point_cloud
from point import Point
from settings import *
from tracker import track_features

def main():
    video_capture = cv2.VideoCapture(video_path)

    ret, first_frame = video_capture.read()
    if not ret:
        print("Failed to capture video")
        return
    
    global frame_shape, frame_height, frame_width
    frame_shape = first_frame.shape
    frame_height, frame_width = frame_shape[:2]
    
    first_frame_grayscale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    prev_points = pick_points(first_frame)
    track_mask = np.zeros_like(first_frame)

    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            break
        
        current_frame_grayscale = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        curr_points = track_features(first_frame_grayscale, current_frame_grayscale, prev_points)

        map_points(curr_points)

        plot_point_cloud(curr_points)

        current_frame, track_mask = draw_tracks(current_frame, track_mask, curr_points)
        
        output_frame = cv2.add(current_frame, track_mask)
        cv2.imshow('Tracking Frame', output_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        
        first_frame_grayscale = current_frame_grayscale.copy()
        prev_points = curr_points

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
