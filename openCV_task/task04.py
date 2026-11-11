import cv2
import sys

def show_fit(name, img, max_w=600):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

img = cv2.imread("../data/test.jpg")

if img is None:
    print("Error: failed to open image")
    sys.exit()

b, g, r = cv2.split(img)
swapped = cv2.merge([r, g, b])

show_fit("original", img)
show_fit("after", swapped)
cv2.waitKey(0)
cv2.destroyAllWindows()