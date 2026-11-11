import ctypes
import cv2
import glfw
import numpy as np
from OpenGL.GL import *

# ---------- 1. Shaders ----------
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
out vec2 UV;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    UV = uv;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D camTex;
void main() {
    color = texture(camTex, UV).rgb;
}
"""


def compile_shader(src, kind):
    shader = glCreateShader(kind)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(
            glGetShaderInfoLog(shader).decode())
    return shader


def load_shaders(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        raise RuntimeError(
            glGetProgramInfoLog(program).decode())
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program


# ---------- 2. Camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    raise SystemExit
ret, frame = cap.read()
if not ret:
    print("Error: cannot read frame")
    raise SystemExit
h, w = frame.shape[:2]
print("frame size:", w, "x", h)

# ---------- 3. Window ----------
if not glfw.init():
    raise SystemExit("Failed to init GLFW")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE,
                 glfw.OPENGL_CORE_PROFILE)
window = glfw.create_window(
    w, h, "Step 3 (ESC: quit)", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Failed to create window")
glfw.make_context_current(window)

program = load_shaders(VERTEX_SHADER,
                       FRAGMENT_SHADER)
tex_loc = glGetUniformLocation(program, "camTex")

# ---------- 4. Full-screen quad ----------
#   x,    y,    u,   v
quad = np.array([
    -1.0, -1.0, 0.0, 1.0,   # bottom-left
     1.0, -1.0, 1.0, 1.0,   # bottom-right
     1.0,  1.0, 1.0, 0.0,   # top-right
    -1.0, -1.0, 0.0, 1.0,   # bottom-left
     1.0,  1.0, 1.0, 0.0,   # top-right
    -1.0,  1.0, 0.0, 0.0,   # top-left
], dtype=np.float32)

vao = glGenVertexArrays(1)
glBindVertexArray(vao)

vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, quad.nbytes,
             quad, GL_STATIC_DRAW)

stride = 4 * 4
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                      stride, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                      stride, ctypes.c_void_p(2 * 4))

# ---------- 5. Texture ----------
tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, tex)
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
             GL_BGR, GL_UNSIGNED_BYTE, None)

glClearColor(0.0, 0.0, 0.4, 0.0)

# ---------- 6. Main loop ----------
while not glfw.window_should_close(window):
    ret, frame = cap.read()
    if not ret:
        break

    glBindTexture(GL_TEXTURE_2D, tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                    GL_BGR, GL_UNSIGNED_BYTE, frame)

    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glUniform1i(tex_loc, 0)
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)

    glfw.swap_buffers(window)
    glfw.poll_events()
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break

# ---------- 7. Cleanup ----------
glDeleteBuffers(1, [vbo])
glDeleteTextures(1, [tex])
glDeleteVertexArrays(1, [vao])
glDeleteProgram(program)
cap.release()
glfw.terminate()