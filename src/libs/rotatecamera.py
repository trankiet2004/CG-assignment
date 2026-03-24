import math
import numpy as np
from libs.transform import lookat, perspective

class RotateCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), radius=3.0, height=1.2, period=8.0):
        self.target = np.array(target, dtype=np.float32)
        self.radius = float(radius)
        self.height = float(height)
        self.period = float(period)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        self.eye = np.array([0.0, height, radius], dtype=np.float32)
        self.forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def update(self, t):
        # Bước 1: tính góc quay theo thời gian
        theta = 2.0 * math.pi * (t / self.period)

        # Bước 2: rotate hướng nhìn quanh trục Y
        self.forward = np.array([
            -math.sin(theta),
            0.0,
            -math.cos(theta)
        ], dtype=np.float32)

        # chuẩn hoá cho chắc
        self.forward = self.forward / np.linalg.norm(self.forward)

        # Bước 3: từ target lùi ra sau theo hướng ngược với hướng nhìn
        self.eye = self.target - self.forward * self.radius
        self.eye[1] += self.height

    def view_matrix(self):
        return lookat(self.eye, self.target, self.up)

    def projection_matrix(self, win_size):
        aspect = win_size[0] / win_size[1]
        return perspective(35.0, aspect, 0.1, 100.0)

    def drag(self, old, new, win_size):
        pass

    def pan(self, old, new):
        pass

    def zoom(self, delta, size):
        zoom_factor = 1.0 - 0.05 * delta
        self.radius = max(0.2, self.radius * zoom_factor)
