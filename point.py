import numpy as np

class Point:

    def __init__(self, position, velocity=(0, 0), color=None):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.color = np.array(color) if color is not None else np.random.randint(0, 255, (3,))

        self.real_position = np.array([0, 0, 0])
        

    def update(self, new_position):

        delta = new_position - self.position
        new_velocity = delta * .1 + self.velocity * .9

        return Point(new_position, velocity=new_velocity, color=self.color)