import cv2

from plotter import Plotter
from settings import *

Plotter(cv2.VideoCapture(video_path))