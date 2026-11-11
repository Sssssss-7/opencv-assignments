import cv2
import numpy as np

# ---------- 1. Load calibration ----------
data = np.load("../output/calib_webcam.npz")
K = data["K"]
dist = data["dist"]
pattern_size = tuple(
    int(v) for v in data["pattern_size"])
square_size = float(data["square_size"])
print("K:\n", K)
print("pattern:", pattern_size,
      "square:", square_size, "cm")

# ---------- 2. 3D corners on the board ----------
objp = np.zeros(
    (pattern_size[0] * pattern_size[1], 3),
    np.float32)
objp[:, :2] = np.mgrid[
    0:pattern_size[0],
    0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30, 0.001)
axis_length = 3 * square_size

# ---------- 3. Open camera ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    raise SystemExit

# ---------- 4. Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame,
                        cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
            gray, pattern_size, None,
            cv2.CALIB_CB_FAST_CHECK)

    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11),
            (-1, -1), criteria)
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners, K, dist)
        if ok:
            cv2.drawFrameAxes(
                frame, K, dist, rvec, tvec,
                axis_length, 3)
            z_cm = tvec[2][0]
            cv2.putText(frame,
                f"Z = {z_cm:.1f} cm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("Step 2: pose (Q: quit)",
               frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("frame size:", frame.shape)
cap.release()
cv2.destroyAllWindows()