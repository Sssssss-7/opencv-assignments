import cv2
import sys

img = cv2.imread("../data/test_a.jpg")
if img is None:
    print("Error")
    sys.exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

max_w, max_h = 1200, 800
h, w = gray.shape[:2]
scale = min(max_w / w, max_h / h, 1.0)
gray = cv2.resize(gray, None, fx=scale, fy=scale)

def on_trackbar(value):
    ret, binary = cv2.threshold(gray, value, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", binary)

cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Threshold", "Binary", 128, 255, on_trackbar )

on_trackbar(128)

cv2.waitKey(0)
cv2.destroyAllWindows()


