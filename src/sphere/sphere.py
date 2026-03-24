import sys
import os

# Add parent directory to path to import libs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libs.shader import *
from libs import transform as T
from libs.buffer import *
from libs.lighting import LightingManager
import ctypes
import glfw
import numpy as np
import OpenGL.GL as GL


class Sphere(object):
    def __init__(self, vert_shader, frag_shader, subdivision_level=4):
        self.vert_shader = vert_shader
        self.frag_shader = frag_shader
        self.subdivision_level = subdivision_level
        self.base_vertices = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.942809, -0.33333],
                [-0.816497, -0.471405, -0.33333],
                [0.816497, -0.471405, -0.33333]
            ],
            dtype=np.float32
        )
        # self.indices = np.array(
        #     [0, 1, 2, 3, 4, 5, 0, 1, 1, 0, 0, 2, 4, 4, 1, 1, 5, 3],
        #     dtype=np.int32
        # )

        # self.base_faces = np.array(
        #     [
        #         [0, 1, 2],
        #         [0, 2, 3],
        #         [0, 3, 1],
        #         [1, 3, 2],
        #     ],
        #     dtype=np.int32
        # )
        
        self.normals = self.base_vertices.copy()
        self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

        # colors: RGB format
        self.colors = np.array(
            [  # R    G    B
                [1.0, 0.0, 0.0],  # A <= Bottom:
                [0.0, 1.0, 0.0],  # B
                [0.0, 0.0, 1.0],  # C
                [1.0, 1.0, 0.0],  # D
            ],
            dtype=np.float32
        )

        self.vao = VAO()

        self.shader = Shader(vert_shader, frag_shader)
        self.uma = UManager(self.shader)
        self.lighting = LightingManager(self.uma)
        
    #helper 
    @staticmethod
    def _normalize(v):
        n = np.linalg.norm(v)
        if n < 1e-12:
            return v
        return v / n

    def _subdivide_triangle(self, A, B, C, cA, cB, cC, level, out_pos, out_col):
        if level <= 0:
            out_pos.append(A); out_pos.append(B); out_pos.append(C)
            out_col.append(cA); out_col.append(cB); out_col.append(cC)
            return

        # Midpoints in Euclidean space (CHƯA normalize)
        Pab = (A + B) * 0.5
        Pbc = (B + C) * 0.5
        Pca = (C + A) * 0.5

        # Colors at midpoints using AREA barycentric on parent triangle ABC
        cab = self._interp_color_area(A, B, C, cA, cB, cC, Pab)
        cbc = self._interp_color_area(A, B, C, cA, cB, cC, Pbc)
        cca = self._interp_color_area(A, B, C, cA, cB, cC, Pca)

        # Now project positions onto sphere
        Pab = self._normalize(Pab)
        Pbc = self._normalize(Pbc)
        Pca = self._normalize(Pca)

        self._subdivide_triangle(A,   Pab, Pca, cA,  cab, cca, level - 1, out_pos, out_col)
        self._subdivide_triangle(B,   Pbc, Pab, cB,  cbc, cab, level - 1, out_pos, out_col)
        self._subdivide_triangle(C,   Pca, Pbc, cC,  cca, cbc, level - 1, out_pos, out_col)
        self._subdivide_triangle(Pab, Pbc, Pca, cab, cbc, cca, level - 1, out_pos, out_col)

    
    def _barycentric_area(self, A, B, C, P):
        # Oriented normal of triangle ABC
        n = np.cross(B - A, C - A)
        denom = np.dot(n, n)
        if denom < 1e-12:
            # triangle degenerate
            return 1.0, 0.0, 0.0

        # Oriented double-areas via dot(n, cross(...))
        wA = np.dot(n, np.cross(B - P, C - P)) / denom
        wB = np.dot(n, np.cross(C - P, A - P)) / denom
        wC = np.dot(n, np.cross(A - P, B - P)) / denom

        # small numeric drift fix
        s = wA + wB + wC
        if abs(s) > 1e-12:
            wA, wB, wC = wA / s, wB / s, wC / s
        return wA, wB, wC

    def _interp_color_area(self, A, B, C, cA, cB, cC, P):
        wA, wB, wC = self._barycentric_area(A, B, C, P)
        return wA * cA + wB * cB + wC * cC

    def _build_geometry(self):
        # base verts & colors
        V = np.array([self._normalize(v) for v in self.base_vertices], dtype=np.float32)
        C = np.array(self.colors, dtype=np.float32)

        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [0, 3, 1],
                [1, 3, 2],
            ],
            dtype=np.int32
        )

        out_pos, out_col = [], []
        for i, j, k in faces:
            self._subdivide_triangle(
                V[i], V[j], V[k],
                C[i], C[j], C[k],
                self.subdivision_level,
                out_pos, out_col
            )

        self.vertices = np.array(out_pos, dtype=np.float32)
        self.colors   = np.array(out_col, dtype=np.float32)

        # normals for lighting
        self.normals = self.vertices / np.linalg.norm(self.vertices, axis=1, keepdims=True)

        # EBO 0..N-1 (giữ pipeline VAO/EBO của bạn)
        self.indices = np.arange(self.vertices.shape[0], dtype=np.uint32)

    """
    Create object -> call setup -> call draw
    """
    def setup(self):
        
        self._build_geometry()

        # setup VAO for drawing prism
        self.vao.add_vbo(0, self.vertices, ncomponents=3, stride=0, offset=None)
        self.vao.add_vbo(1, self.colors, ncomponents=3, stride=0, offset=None)
        
        # Add normals for Gouraud/Phong shading (if shader needs it)
        if 'gouraud' in self.vert_shader.lower() or 'phong' in self.vert_shader.lower():
            self.vao.add_vbo(2, self.normals, ncomponents=3, stride=0, offset=None)

        # setup EBO for drawing prism
        self.vao.add_ebo(self.indices)

        return self

    def draw(self, projection, view, model):
        GL.glUseProgram(self.shader.render_idx)
        modelview = view

        self.uma.upload_uniform_matrix4fv(projection, 'projection', True)
        self.uma.upload_uniform_matrix4fv(modelview, 'modelview', True)
        
        # Setup lighting if using Gouraud or Phong shader
        if 'gouraud' in self.vert_shader.lower():
            self.lighting.setup_gouraud()
        elif 'phong' in self.vert_shader.lower():
            self.lighting.setup_phong(mode=1)

        self.vao.activate()
        GL.glDrawElements(GL.GL_TRIANGLE_STRIP, self.indices.shape[0], GL.GL_UNSIGNED_INT, None)


    def key_handler(self, key):

        if key == glfw.KEY_1:
            self.selected_texture = 1
        if key == glfw.KEY_2:
            self.selected_texture = 2

