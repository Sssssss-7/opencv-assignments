import glfw
from OpenGL.GL import *

if not glfw.init():
    raise SystemExit("Failed to init GLFW")

window = glfw.create_window(
    640, 480, "Step 1", None, None)
glfw.make_context_current(window)

while not glfw.window_should_close(window):
    glClearColor(0.0, 0.0, 0.4, 0.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()