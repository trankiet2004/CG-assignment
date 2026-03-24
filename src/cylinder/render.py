import OpenGL.GL as GL
import glfw
from itertools import cycle
import sys
import os

# path setup
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sphere import Sphere
from libs.rotatecamera import RotateCamera


class RenderEngine:
    """GLFW render window + render loop"""

    def __init__(self, width=800, height=800, title="Render"):
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        glfw.window_hint(glfw.DEPTH_BITS, 16)
        glfw.window_hint(glfw.DOUBLEBUFFER, True)

        self.win = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.win)

        self.camera = RotateCamera(
            target=(0.0, 0.0, 0.0),
            radius=6.0,
            height=1.2,
            period=8.0
        )

        self.mouse = (0, 0)

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

        GL.glClearColor(0.5, 0.5, 0.5, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)

        self.drawables = []

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
    engine = RenderEngine()

    model = Sphere("./color_interp.vert", "./color_interp.frag").setup()
    engine.add(model)

    engine.run()


if __name__ == "__main__":
    glfw.init()
    main()
    glfw.terminate()