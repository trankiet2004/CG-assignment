import OpenGL.GL as GL
import glfw
from itertools import cycle
import sys
import os

# =========================
# Path setup
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# =========================
# Imports project modules
# =========================
from libs.transform import Trackball
from libs.rotatecamera import RotateCamera

# Chọn object muốn render
from sphere.sphere import Sphere
# from sphere import Sphere


class DualCamera:
    """
    Quản lý 2 camera:
    - Trackball: camera mặc định, điều khiển bằng chuột
    - RotateCamera: camera tự xoay quanh object
    """

    MODE_TRACKBALL = 0
    MODE_ROTATE = 1

    def __init__(
        self,
        rotate_target=(0.0, 0.0, 0.0),
        rotate_radius=3.0,
        rotate_height=1.2,
        rotate_period=8.0,
        start_mode="trackball",
    ):
        self.trackball = Trackball()

        self.rotate_camera = RotateCamera(
            target=rotate_target,
            radius=rotate_radius,
            height=rotate_height,
            period=rotate_period,
        )

        if start_mode.lower() == "rotate":
            self.mode = self.MODE_ROTATE
            self.active_camera = self.rotate_camera
        else:
            self.mode = self.MODE_TRACKBALL
            self.active_camera = self.trackball

    def use_trackball(self):
        self.mode = self.MODE_TRACKBALL
        self.active_camera = self.trackball

    def use_rotate(self):
        self.mode = self.MODE_ROTATE
        self.active_camera = self.rotate_camera

    def toggle(self):
        if self.mode == self.MODE_TRACKBALL:
            self.use_rotate()
        else:
            self.use_trackball()

    def update(self, t):
        if self.mode == self.MODE_ROTATE:
            self.rotate_camera.update(t)

    def view_matrix(self):
        return self.active_camera.view_matrix()

    def projection_matrix(self, win_size):
        return self.active_camera.projection_matrix(win_size)

    def drag(self, old, new, win_size):
        if self.mode == self.MODE_TRACKBALL:
            self.trackball.drag(old, new, win_size)

    def pan(self, old, new):
        if self.mode == self.MODE_TRACKBALL:
            self.trackball.pan(old, new)

    def zoom(self, delta, size):
        self.active_camera.zoom(delta, size)

    def is_trackball(self):
        return self.mode == self.MODE_TRACKBALL

    def is_rotate(self):
        return self.mode == self.MODE_ROTATE

    def get_active_name(self):
        if self.mode == self.MODE_TRACKBALL:
            return "Trackball"
        return "RotateCamera"


class Viewdual:
    """GLFW viewer dùng DualCamera."""

    def __init__(self, width=800, height=800, title="Dual Camera Viewer"):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, title, None, None)
        if not self.win:
            raise RuntimeError("Cannot create GLFW window")

        glfw.make_context_current(self.win)

        self.camera = DualCamera(
            rotate_target=(0.0, 0.0, 0.0),
            rotate_radius=6.0,
            rotate_height=1.2,
            rotate_period=8.0,
            start_mode="trackball",
        )

        self.mouse = (0, 0)
        self.drawables = []

        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_cursor_pos_callback(self.win, self.on_mouse_move)
        glfw.set_scroll_callback(self.win, self.on_scroll)

        print(
            "OpenGL",
            GL.glGetString(GL.GL_VERSION).decode(),
            "| GLSL",
            GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode(),
            "| Renderer",
            GL.glGetString(GL.GL_RENDERER).decode(),
        )
        print("Press Space to switch camera")
        print("Current camera:", self.camera.get_active_name())

        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

    def add(self, *drawables):
        self.drawables.extend(drawables)

    def run(self):
        while not glfw.window_should_close(self.win):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            win_size = glfw.get_window_size(self.win)

            self.camera.update(glfw.get_time())

            view = self.camera.view_matrix()
            projection = self.camera.projection_matrix(win_size)

            for drawable in self.drawables:
                drawable.draw(projection, view, None)

            glfw.swap_buffers(self.win)
            glfw.poll_events()

    def on_key(self, _win, key, _scancode, action, _mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)

            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))

            if key == glfw.KEY_SPACE and action == glfw.PRESS:
                self.camera.toggle()
                print("Switched to", self.camera.get_active_name())

            for drawable in self.drawables:
                if hasattr(drawable, "key_handler"):
                    drawable.key_handler(key)

    def on_mouse_move(self, win, xpos, ypos):
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)

        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.camera.drag(old, self.mouse, glfw.get_window_size(win))

        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.camera.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        self.camera.zoom(deltay, glfw.get_window_size(win)[1])


def main():
    viewer = Viewdual()

    vert_path = os.path.join(current_dir, "color_interp.vert")
    frag_path = os.path.join(current_dir, "color_interp.frag")

    model = Sphere(vert_path, frag_path).setup()
    # model = Sphere(vert_path, frag_path).setup()

    viewer.add(model)
    viewer.run()


if __name__ == "__main__":
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    try:
        main()
    finally:
        glfw.terminate()