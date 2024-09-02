import cv2

video_name = "road"

video_path = f"./videos/{video_name}.mp4"

num_chunks_x = 80
num_chunks_y = 60
num_points_per_chunk = 100

num_points = num_points_per_chunk * num_chunks_x * num_chunks_y

focal_length = 1

video = cv2.VideoCapture(video_path)

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

video.release()
