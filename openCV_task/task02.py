import cv2
import sys

cap = cv2.VideoCapture("../data/test.mp4")

if not cap.isOpened():
    print("Error: failed to open video")
    sys.exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("video", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


