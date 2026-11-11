import cv2
import sys
import os

cap = cv2.VideoCapture("../data/test.mp4")

if not cap.isOpened():
    print("Error: failed to open video")
    sys.exit()

os.makedirs("../output", exist_ok = True)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    name = f"../output/frame_{count:04d}.png"
    cv2.imwrite(name, frame)
    count += 1

    cv2.imshow("video", frame)

    if cv2.waitKey(30) & 0XFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

