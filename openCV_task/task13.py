import cv2
import sys
import numpy as np
import os

def show_fit(name, img, max_w = 1000):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx = scale, fy = scale))

calib_path = "../output/calib.npz"
if not os.path.exists(calib_path):
    print("Error: calib.npz not found")
    sys.exit()

data = np.load(calib_path)
K = data["K"]
dist = data["dist"]
rvecs = data["rvecs"]
tvecs = data["tvecs"]
files = data["files"]
pattern_size = (int(data["pattern_size"][0]), int(data["pattern_size"][1]))
square_size = float(data["square_size"])

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

show_images = True
errors_all = []

for i, fname in enumerate(files):
    img = cv2.imread(str(fname))
    if img is None:
        print(f"{fname}: could not be read, skipped")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        print(f"{fname}: corners NOT found, skipped")
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    projected,_ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)

    detected_2d = corners.reshape(-1, 2)
    projected_2d = projected.reshape(-1, 2)
    distances = np.linalg.norm(detected_2d - projected_2d, axis=1)
    mean_error = distances.mean()
    errors_all.append(mean_error)
    print(f"{os.path.basename(str(fname))}: mean reprojection error = {mean_error:.4f}px")

    if show_images:
        vis = img.copy()
        for p in detected_2d:
            cv2.circle(vis, (int(p[0]), int(p[1])), 6, (0,255,0), 1)
        for p in projected_2d:
            cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0,0,255), -1)
        cv2.putText(vis, f"Mean error:{mean_error:.3f}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        show_fit("Reprojection(SPACE: next, Q: quit)", vis)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(" "):
                break
            elif key == ord('q'):
                show_images = False
                cv2.destroyAllWindows()
                break
cv2.destroyAllWindows()
if len(errors_all) > 0:
    print(f"\nAverage over {len(errors_all)} images: {np.mean(errors_all):.4f} px")


