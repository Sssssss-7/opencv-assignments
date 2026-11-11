import cv2
import sys
import numpy as np

# Load the image and convert to grayscale
img = cv2.imread("../data/test_a.jpg")
if img is None:
    print("Error: failed to open image")
    sys.exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

max_w, max_h = 800, 800
h, w = gray.shape
scale = min(max_w / w, max_h / h, 1.0)
gray = cv2.resize(gray, None, fx=scale, fy=scale)

# Binarize with Otsu's method (automatic threshold)
ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {ret:.0f}")

# Label connected components with 8-connectivity
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Components (excluding background): {num_labels - 1}")

# Assign a random color to each label; the background stays black
np.random.seed(0)
colors = np.random.randint(50, 255, (num_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]
colored = colors[labels]

# Draw a bounding box and centroid for each object, skipping tiny ones
min_area = 100
count = 0
for i in range(1, num_labels):
    x, y, bw, bh, area = stats[i]
    if area < min_area:
        continue
    cv2.rectangle(colored, (int(x), int(y)), (int(x + bw), int(y + bh)), (255, 255, 255), 2)
    cx, cy = centroids[i]
    cv2.circle(colored, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    count += 1
print(f"Objects with area >= {min_area}: {count}")

# Show the binary image and the labeled image side by side
binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
grid = np.hstack([binary_bgr, colored])
cv2.imshow("Binary | Labeled", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()