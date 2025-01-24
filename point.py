import numpy as np

from utils import update_weighted_time_series

class Point:

    def __init__(self, position_2d, color=None):
        self.position_2d = np.array(position_2d)
        self.velocity_2d = None

        self.position_3d = None
        self.velocity_3d = None

        self.color = np.array(color) if color is not None else np.random.randint(0, 255, (3,))

    def update_2d(self, new_position_2d):

        new_velocity_2d = new_position_2d - self.position_2d
        self.velocity_2d = update_weighted_time_series(self.velocity_2d, new_velocity_2d, .5)
        self.position_2d = new_position_2d
    
    def update_3d(self, new_position_3d):

        self.position_3d = update_weighted_time_series(self.position_3d, new_position_3d, 1)
        new_velocity_3d = new_position_3d - self.position_3d
        self.velocity_3d = update_weighted_time_series(self.velocity_3d, new_velocity_3d, .5)