import cv2
import numpy as np

cap = cv2.VideoCapture("ball.mp4")

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

ret, old_frame = cap.read()

hsv = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)

contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

c = max(contours, key=cv2.contourArea)
((x, y), radius) = cv2.minEnclosingCircle(c)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(old_frame)

p0 = np.array([[[x, y]]], dtype=np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    img = cv2.add(frame, mask)

    cv2.imshow("Optical Flow - Object Tracking", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
