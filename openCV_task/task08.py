import cv2
import sys
import os
import numpy as np

def show_fit(name, img, max_w=900,max_h=1000):
    scale = min(max_h / img.shape[0],  max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

img = cv2.imread("../data/test.jpg")
if img is None:
    print("Error: failed to open image")
    sys.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

# Fix the random seed so the noise is the same every run
np.random.seed(0)

# --- Gaussian noise ---
sigma = 25
noise = np.random.normal(0, sigma, (h, w))
gaussian = gray.astype(np.float64) + noise
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

# --- Salt-and-pepper noise ---
amount = 0.05
sp = gray.copy()
rand = np.random.rand(h,w)
sp[rand < amount / 2] = 0
sp[rand > 1 - amount / 2] = 255

os.makedirs("../output", exist_ok = True)
cv2.imwrite("../output/noise_gaussian.png", gaussian)
cv2.imwrite("../output/noise_sp.png", sp)

grid = np.hstack([gray, gaussian, sp])

show_fit("Original | Gaussian | Salt-and-pepper", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()


