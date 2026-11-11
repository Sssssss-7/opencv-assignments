import ctypes
import cv2
import glfw
import numpy as np
from OpenGL.GL import *

# ============ 1. Shaders ============
BG_VS = """
#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
out vec2 UV;
void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
    UV = uv;
}
"""

BG_FS = """
#version 330 core
in vec2 UV;
out vec3 color;
uniform sampler2D camTex;
void main() {
    color = texture(camTex, UV).rgb;
}
"""

CUBE_VS = """
#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 col;
uniform mat4 MVP;
out vec3 fragColor;
void main() {
    gl_Position = MVP * vec4(pos, 1.0);
    fragColor = col;
}
"""

CUBE_FS = """
#version 330 core
in vec3 fragColor;
out vec3 color;
void main() {
    color = fragColor;
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


# ============ 2. Matrices  ============
def projection_from_K(K, w, h, near=1.0, far=500.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * fx / w
    P[0, 2] = 1 - 2 * cx / w
    P[1, 1] = 2 * fy / h
    P[1, 2] = 2 * cy / h - 1
    P[2, 2] = -(far + near) / (far - near)
    P[2, 3] = -2 * far * near / (far - near)
    P[3, 2] = -1
    return P


def view_from_pose(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = tvec.ravel()
    cv_to_gl = np.diag([1, -1, -1, 1]).astype(np.float32)
    return cv_to_gl @ Rt


def translate(x, y, z):
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = [x, y, z]
    return T


# ============ 3. Cube data (NEW) ============
def make_cube(s):
    a = s / 2
    # x, y in [-a, a]; z from 0 (on board) to -s (toward camera)
    c = [(-a, -a, 0), (a, -a, 0), (a, a, 0), (-a, a, 0),
         (-a, -a, -s), (a, -a, -s), (a, a, -s), (-a, a, -s)]
    faces = [
        ((0, 1, 2, 3), (0.5, 0.5, 0.5)),  # bottom: gray
        ((4, 5, 6, 7), (1.0, 0.0, 0.0)),  # top: red
        ((0, 1, 5, 4), (0.0, 1.0, 0.0)),  # side: green
        ((3, 2, 6, 7), (0.0, 0.0, 1.0)),  # side: blue
        ((0, 3, 7, 4), (1.0, 1.0, 0.0)),  # side: yellow
        ((1, 2, 6, 5), (1.0, 0.0, 1.0)),  # side: magenta
    ]
    data = []
    for (i0, i1, i2, i3), color in faces:
        for i in (i0, i1, i2, i0, i2, i3):
            data.extend(c[i])
            data.extend(color)
    return np.array(data, dtype=np.float32)


# ============ 4. Calibration + board ============
calib = np.load("../output/calib_webcam.npz")
K = calib["K"]
dist = calib["dist"]
pattern_size = tuple(int(v) for v in calib["pattern_size"])
square_size = float(calib["square_size"])

objp = np.zeros(
    (pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:pattern_size[0],
    0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ============ 5. Camera ============
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

# ============ 6. Window ============
if not glfw.init():
    raise SystemExit("Failed to init GLFW")
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE,
                 glfw.OPENGL_CORE_PROFILE)
window = glfw.create_window(
    w, h, "Marker AR (ESC: quit)", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Failed to create window")
glfw.make_context_current(window)

bg_program = load_shaders(BG_VS, BG_FS)
tex_loc = glGetUniformLocation(bg_program, "camTex")
cube_program = load_shaders(CUBE_VS, CUBE_FS)
mvp_loc = glGetUniformLocation(cube_program, "MVP")

# ============ 7. Background quad (same as Step 3) ============
quad = np.array([
    -1.0, -1.0, 0.0, 1.0,
     1.0, -1.0, 1.0, 1.0,
     1.0,  1.0, 1.0, 0.0,
    -1.0, -1.0, 0.0, 1.0,
     1.0,  1.0, 1.0, 0.0,
    -1.0,  1.0, 0.0, 0.0,
], dtype=np.float32)

bg_vao = glGenVertexArrays(1)
glBindVertexArray(bg_vao)
bg_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, bg_vbo)
glBufferData(GL_ARRAY_BUFFER, quad.nbytes,
             quad, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                      16, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                      16, ctypes.c_void_p(8))

tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, tex)
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER, GL_LINEAR)
glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0,
             GL_BGR, GL_UNSIGNED_BYTE, None)

# ============ 8. Cube VAO + VBO  ============
cube_size = 2 * square_size          # 5 cm
cube = make_cube(cube_size)
cube_count = len(cube) // 6          # 36 vertices

cube_vao = glGenVertexArrays(1)
glBindVertexArray(cube_vao)
cube_vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, cube_vbo)
glBufferData(GL_ARRAY_BUFFER, cube.nbytes,
             cube, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                      24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                      24, ctypes.c_void_p(12))

# ============ 9. Fixed matrices ============
P = projection_from_K(K, w, h)
cx_board = (pattern_size[0] - 1) * square_size / 2
cy_board = (pattern_size[1] - 1) * square_size / 2
M = translate(cx_board, cy_board, 0)

glClearColor(0.0, 0.0, 0.4, 0.0)
glDepthFunc(GL_LESS)

# ============ 10. Main loop ============
while not glfw.window_should_close(window):
    ret, frame = cap.read()
    if not ret:
        break

    # --- (a) pose estimation (same as Step 2) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, None,
        cv2.CALIB_CB_FAST_CHECK)
    pose_ok = False
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        pose_ok, rvec, tvec = cv2.solvePnP(
            objp, corners, K, dist)

    # --- (b) background (same as Step 3) ---
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h,
                    GL_BGR, GL_UNSIGNED_BYTE, frame)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glUseProgram(bg_program)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex)
    glUniform1i(tex_loc, 0)
    glBindVertexArray(bg_vao)
    glDrawArrays(GL_TRIANGLES, 0, 6)

    # --- (c) cube  ---
    if pose_ok:
        V = view_from_pose(rvec, tvec)
        MVP = (P @ V @ M).astype(np.float32)

        glEnable(GL_DEPTH_TEST)
        glUseProgram(cube_program)
        glUniformMatrix4fv(mvp_loc, 1, GL_TRUE, MVP)
        glBindVertexArray(cube_vao)
        glDrawArrays(GL_TRIANGLES, 0, cube_count)

    glfw.swap_buffers(window)
    glfw.poll_events()
    if glfw.get_key(window, glfw.KEY_ESCAPE) \
            == glfw.PRESS:
        break

# ============ 11. Cleanup ============
glDeleteBuffers(2, [bg_vbo, cube_vbo])
glDeleteTextures(1, [tex])
glDeleteVertexArrays(2, [bg_vao, cube_vao])
glDeleteProgram(bg_program)
glDeleteProgram(cube_program)
cap.release()
glfw.terminate()
