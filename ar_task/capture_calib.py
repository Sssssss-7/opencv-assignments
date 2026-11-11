import cv2
import os

# ---------- 1. Setup ----------
save_dir = "../data/webcam_calib"
os.makedirs(save_dir, exist_ok=True)
pattern_size = (9, 6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    raise SystemExit

count = 0
frame = None

# ---------- 2. Main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame")
        break

    gray = cv2.cvtColor(frame,
                        cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, None,
        cv2.CALIB_CB_FAST_CHECK)

    vis = frame.copy()
    if found:
        cv2.drawChessboardCorners(
            vis, pattern_size, corners, found)
    cv2.putText(vis, f"saved: {count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("SPACE: save, Q: quit", vis)

    # ---------- 3. Keys ----------
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if found:
            name = os.path.join(
                save_dir, f"calib_{count:02d}.jpg")
            cv2.imwrite(name, frame)
            print("saved", name)
            count += 1
        else:
            print("corners not found, not saved")
    elif key == ord('q'):
        break

# ---------- 4. Cleanup ----------
if frame is not None:
    print("image size:", frame.shape)
print(f"total saved: {count}")
cap.release()
cv2.destroyAllWindows()