import cv2
import sys
import os
import glob
import numpy as np

def show_fit(name, img, max_w = 1000):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

pattern_size = (9, 6)
square_size = 2.5      # physical   measure

# ---------- 3D coordinates of the corners on the board (Z = 0) ----------
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size


obj_points = []
img_points = []
used_files = []
image_size = None
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

files = sorted(glob.glob("../data/webcam_calib/*.jpg"))
if len(files) == 0:
    print("Error: no image found in calib")
    sys.exit()

show_images = True

for fname in files:
    img = cv2.imread(fname)
    if img is None:
        print(f"{fname}: could not be read, skipped")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not found:
        print(f"{fname}: corners NOT found, skipped")
        continue

    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    obj_points.append(objp)
    img_points.append(corners)
    used_files.append(fname)

    if show_images:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        show_fit("Detected corners(SPACE : next, Q : quit)", vis)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord(' '):
                break
            elif key == ord('q'):
                show_images = False
                cv2.destroyAllWindows()
                break

print(f"Corners found in {len(used_files)} / {len(files)} images")
if len(used_files) <5 :
    print("Error: too few valid images.")
    sys.exit()

rms, K, dist, rvecs, tvecs =cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

print(f"\nRms reprojection error: {rms:.4f} px")
print("\nIntrinsic matrix K:\n", K)  ###
print("\nDistortion coefficients (k1, k2, p1, p2, k3):\n", dist.ravel())

R, _ = cv2.Rodrigues(rvecs[0])
print(f"\nExtrinsics of {used_files[0]}: ")
print("Rotation matrix R:\n", R)
print("Translation vector t (in cm)\n", tvecs[0].ravel())

os.makedirs("../output", exist_ok=True)
np.savez("../output/calib_webcam.npz", K=K, dist=dist,
rvecs = np.array(rvecs), tvecs = np.array(tvecs),
files = np.array(used_files), pattern_size = pattern_size, square_size = square_size)
print("\nSaved to ../output/calib_webcam.npz")









