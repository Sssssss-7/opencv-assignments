import cv2
import sys
import os
import numpy as np


def show_fit(name, img, max_w=1000):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

calib_path = "../output/calib.npz"
if not os.path.exists(calib_path):
    print("Error: calib.npz not found.")
    sys.exit()

data = np.load(calib_path)
K = data["K"]
dist = data["dist"]
files = data["files"]
pattern_size = (int(data["pattern_size"][0]) , int(data["pattern_size"][1]))
square_size = float(data["square_size"])

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

axis_length = 3 * square_size
quit_all = False

for fname in files:
    img = cv2.imread(str(fname))
    if img is None:
        print(f"{fname}: could not be read, skipped")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        print(f"{fname}:corners NOT found, skipped")
        continue
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)
    if not ok:
        print(f"{fname}: solvePnP failed, skipped")
        continue

    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -R.T @ tvec  #in the board coordinate
    distance = np.linalg.norm(tvec)

    name = os.path.basename(str(fname))
    print(f"\n{name}")
    print("  rvec:", rvec.ravel())
    print("  tvec:", tvec.ravel())
    print("  camera position (board coordinates):", camera_pos.ravel())
    print(f"  distance to the board origin: {distance:.2f}")

    vis = img.copy()
    cv2.drawFrameAxes(vis, K, dist, rvec, tvec, axis_length, 3)
    cv2.putText(vis, f"Distance: {distance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    show_fit("Camera pose(SPACE: next, Q: quit)", vis)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(" "):
            break
        elif key == ord("q"):
            quit_all = True
            break

    if quit_all:
        break
cv2.destroyAllWindows()







