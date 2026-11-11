import cv2
import sys

img = cv2.imread("../data/test.jpg")

if img is None:
    print("Error: failed to read")
    sys.exit()

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()