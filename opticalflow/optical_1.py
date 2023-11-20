import cv2
import numpy as np

cap = cv2.VideoCapture("ball.mp4")

pyr_scale = 0.5
levels = 3
winsize = 15
iterations = 3
poly_n = 5
poly_sigma = 1.2
flags = 0

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

mask = np.zeros_like(old_frame)

scale = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        old_gray,
        frame_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    threshold = 1
    magnitude[magnitude < threshold] = 0

    step = 16
    y, x = (
        np.mgrid[step / 2 : frame.shape[0] : step, step / 2 : frame.shape[1] : step]
        .reshape(2, -1)
        .astype(int)
    )
    fx, fy = flow[y, x].T

    lines = np.int32(list(zip(x, y, x + (fx * scale), y + (fy * scale))))
    for x1, y1, x2, y2 in lines:
        cv2.arrowedLine(frame, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

    cv2.imshow("Optical flow", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    old_gray = frame_gray.copy()

cap.release()
cv2.destroyAllWindows()
