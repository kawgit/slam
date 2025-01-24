import cv2

video_name = "road"
video_path = f"./videos/{video_name}.mp4"

num_chunks_x = 160
num_chunks_y = 120
num_points_per_chunk = 10

num_points = num_points_per_chunk * num_chunks_x * num_chunks_y

focal_length = 2

video = cv2.VideoCapture(video_path)

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video.release()
