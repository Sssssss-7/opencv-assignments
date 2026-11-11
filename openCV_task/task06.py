import cv2
import sys

def show_fit(name, img, max_w=600):
    scale = min(max_w / img.shape[1], 1.0)
    cv2.imshow(name, cv2.resize(img, None, fx=scale, fy=scale))

img = cv2.imread("../data/test.jpg")

if img is None:
    print("Error: failed to open image")
    sys.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_fit("original", img)
show_fit("image", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()