import sys
import os
import math
import glfw
import numpy as np
import OpenGL.GL as GL

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs.buffer import *
from libs.lighting import LightingManager


class Cylinder(object):
    def __init__(
        self,
        vert_shader,
        frag_shader,
        radius=0.5,
        height=1.5,
        sectors=36,
    ):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader

        self.radius = float(radius)
        self.height = float(height)
        self.sectors = max(3, int(sectors))

        self.vertices = None
        self.colors = None
        self.normals = None
        self.indices = None

        self.vao = VAO()
        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)

    def _side_color(self, theta, y):
        """
        Tạo màu để nhìn rõ vật thể đang xoay.
        - đổi theo góc theta
        - đổi nhẹ theo chiều cao y
        - thêm stripe xen kẽ để dễ quan sát rotation
        """

        # chuẩn hóa theta về [0, 1]
        t = theta / (2.0 * math.pi)

        # chuẩn hóa y về [0, 1]
        h = (y + self.height * 0.5) / self.height

        # màu cơ bản theo góc
        r = 0.5 + 0.5 * math.cos(theta)
        g = 0.5 + 0.5 * math.sin(theta)
        b = 0.3 + 0.7 * h

        # tạo các stripe để nhìn xoay rõ hơn
        stripe = int((theta / (2.0 * math.pi)) * 12) % 2
        if stripe == 0:
            r *= 1.0
            g *= 0.7
            b *= 0.7
        else:
            r *= 0.7
            g *= 1.0
            b *= 1.0

        return np.array([r, g, b], dtype=np.float32)

    def _top_color(self, x, z):
        """
        Nắp trên: cho màu sáng hơn, có biến thiên theo vị trí
        để nhìn không bị phẳng.
        """
        rr = math.sqrt(x * x + z * z) / max(self.radius, 1e-8)
        return np.array([1.0 - 0.3 * rr, 0.9, 0.2], dtype=np.float32)

    def _bottom_color(self, x, z):
        """
        Nắp dưới: màu khác hẳn nắp trên
        """
        rr = math.sqrt(x * x + z * z) / max(self.radius, 1e-8)
        return np.array([0.2, 0.5 * (1.0 - rr) + 0.2, 1.0 - 0.2 * rr], dtype=np.float32)

    def _build_geometry(self):
        vertices = []
        colors = []
        normals = []
        indices = []

        half_h = self.height * 0.5

        # =============================
        # 1) SIDE SURFACE
        # =============================
        for i in range(self.sectors + 1):
            theta = 2.0 * math.pi * i / self.sectors
            x = self.radius * math.cos(theta)
            z = self.radius * math.sin(theta)

            nx = math.cos(theta)
            nz = math.sin(theta)

            # top vertex
            vertices.append([x, half_h, z])
            colors.append(self._side_color(theta, half_h))
            normals.append([nx, 0.0, nz])

            # bottom vertex
            vertices.append([x, -half_h, z])
            colors.append(self._side_color(theta, -half_h))
            normals.append([nx, 0.0, nz])

        for i in range(self.sectors):
            k1 = 2 * i
            k2 = 2 * i + 1
            k3 = 2 * (i + 1)
            k4 = 2 * (i + 1) + 1

            indices.extend([k1, k2, k3])
            indices.extend([k3, k2, k4])

        # =============================
        # 2) TOP CAP
        # =============================
        top_center_index = len(vertices)
        vertices.append([0.0, half_h, 0.0])
        colors.append(np.array([1.0, 0.95, 0.2], dtype=np.float32))
        normals.append([0.0, 1.0, 0.0])

        top_ring_start = len(vertices)

        for i in range(self.sectors):
            theta = 2.0 * math.pi * i / self.sectors
            x = self.radius * math.cos(theta)
            z = self.radius * math.sin(theta)

            vertices.append([x, half_h, z])
            colors.append(self._top_color(x, z))
            normals.append([0.0, 1.0, 0.0])

        for i in range(self.sectors):
            curr = top_ring_start + i
            nxt = top_ring_start + ((i + 1) % self.sectors)
            indices.extend([top_center_index, curr, nxt])

        # =============================
        # 3) BOTTOM CAP
        # =============================
        bottom_center_index = len(vertices)
        vertices.append([0.0, -half_h, 0.0])
        colors.append(np.array([0.1, 0.3, 1.0], dtype=np.float32))
        normals.append([0.0, -1.0, 0.0])

        bottom_ring_start = len(vertices)

        for i in range(self.sectors):
            theta = 2.0 * math.pi * i / self.sectors
            x = self.radius * math.cos(theta)
            z = self.radius * math.sin(theta)

            vertices.append([x, -half_h, z])
            colors.append(self._bottom_color(x, z))
            normals.append([0.0, -1.0, 0.0])

        # đảo winding cho mặt dưới
        for i in range(self.sectors):
            curr = bottom_ring_start + i
            nxt = bottom_ring_start + ((i + 1) % self.sectors)
            indices.extend([bottom_center_index, nxt, curr])

        self.vertices = np.array(vertices, dtype=np.float32)
        self.colors = np.array(colors, dtype=np.float32)
        self.normals = np.array(normals, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)

    def setup(self):
        self._build_geometry()

        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)

        if "gouraud" in self.vert_shader.lower() or "phong" in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)

        self.vao.add_ebo(self.indices)
        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)

        if model is None:
            model = np.identity(4, dtype=np.float32)

        modelview = view @ model

        self.uma.upload_uniform_matrix4fv(projection, "projection", True)
        self.uma.upload_uniform_matrix4fv(modelview, "modelview", True)

        if "gouraud" in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif "phong" in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLES, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)
        self.vao.deactivate()

    def key_handler(self, key):
        pass