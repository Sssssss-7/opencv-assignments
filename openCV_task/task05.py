import cv2
import sys

def show_fit(name, img, max_w=600):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

img = cv2.imread("../data/test.jpg")

if img is None:
    print("Error: failed to open image")
    sys.exit()

h, w = img.shape[:2]
center = (w/2, h/2)
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w,h))
show_fit("image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()