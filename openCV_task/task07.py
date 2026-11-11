import cv2
import sys
import numpy as np

img = cv2.imread("../data/test.jpg")
if img is None:
    print("Error: failed to open image")
    sys.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Resize to 50% so the comparison grid fits on the screen
gray = cv2.resize(gray, None, fx=0.3, fy=0.3)

# Sobel: first derivative in the x and y directions
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# Laplacian: second derivative
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)

# Take the absolute value and convert back to uint8 (0-255) for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)
lap = cv2.convertScaleAbs(lap)


# Combine Sobel x and y (50% each)
sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

# Canny with low and high thresholds
canny = cv2.Canny(gray, 100, 200)

# Label each image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(gray, "Gray", (10,30), font, 1, 255, 2)
cv2.putText(sobel_x, "Sobel_x", (10,30), font, 1, 255, 2)
cv2.putText(sobel_y, "Sobel Y", (10, 30), font, 1, 255, 2)
cv2.putText(sobel, "Sobel X+Y", (10, 30), font, 1, 255, 2)
cv2.putText(lap, "Laplacian", (10, 30), font, 1, 255, 2)
cv2.putText(canny, "Canny", (10, 30), font, 1, 255, 2)

# Build a 2x3 comparison grid
top = np.hstack([gray, sobel_x, sobel_y])
bottom = np.hstack([sobel, lap, canny])
grid = np.vstack([top, bottom])

cv2.imshow("Edge detection comparison", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()