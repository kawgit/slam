import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_point_cloud(points, z_threshold=3, point_size=10):
    real_positions = np.array([point.real_position for point in points])
    colors = np.array([point.color for point in points]) / 255

    mean = np.mean(real_positions, axis=0)
    std_dev = np.std(real_positions, axis=0)
    z_scores = np.abs((real_positions - mean) / std_dev)
    mask = (z_scores < z_threshold).all(axis=1)
    real_positions = real_positions[mask]
    colors = colors[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp')
    ax.scatter(real_positions[:, 0], real_positions[:, 2], -real_positions[:, 1], c=colors, s=point_size)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    max_range_xz = np.ptp(real_positions[:, [0, 2]], axis=0).max() / 2.0

    mid_x = np.mean(real_positions[:, 0])
    mid_z = np.mean(-real_positions[:, 1])

    ax.set_xlim(mid_x - max_range_xz, mid_x + max_range_xz)
    ax.set_zlim(mid_z - max_range_xz, mid_z + max_range_xz)

    plt.show()


def draw_tracks(frame, track_mask, points):
    for i, point in enumerate(points):

        pt0 = tuple((point.position - point.velocity).astype(np.int32))
        pt1 = tuple((point.position).astype(np.int32))

        track_mask = cv2.line(track_mask, pt0, pt1, (255 - point.color).tolist(), 2)
        
        frame = cv2.circle(frame, pt1, 5, point.color.tolist(), -1)
    
    return frame, track_mask
