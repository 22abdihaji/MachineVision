import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO("yolov8n.pt")  # or yolov8s.pt if you want better accuracy

# Load video
cap = cv2.VideoCapture("IMG_9397.mp4")
if not cap.isOpened():
    print("Error opening video")
    exit()

# Buffer for ball trajectory
ball_positions = deque(maxlen=50)  # Store last 50 positions
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame, conf=0.4)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names

    players = []
    ball = None

    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls_id = det
        label = class_names[int(cls_id)]

        if label == 'person':
            players.append((int(xmin), int(ymin), int(xmax), int(ymax)))
        elif label == 'sports ball':
            ball = (int(xmin), int(ymin), int(xmax), int(ymax))

    # Draw and track
    if ball:
        cx = int((ball[0] + ball[2]) / 2)
        cy = int((ball[1] + ball[3]) / 2)
        ball_positions.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    for (x1, y1, x2, y2) in players:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw ball path
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (0, 0, 255), 2)

    cv2.imshow("Futsal Detection", frame)
    if cv2.waitKey(30) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
