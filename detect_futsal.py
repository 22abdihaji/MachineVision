import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Parameters
FRAME_SKIP = 2
YOLO_SIZE = 640

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
model.fuse()

# Open video
cap = cv2.VideoCapture("futsal vid.mp4")
if not cap.isOpened():
    print("Error opening video")
    exit()

# Drawing and tracking
font = cv2.FONT_HERSHEY_SIMPLEX
ball_positions = deque(maxlen=50)
frame_index = 0

# Define field zones
goal_left = {"x_min": 100, "x_max": 150, "y_min": 250, "y_max": 600}
goal_right = {"x_min": 1100, "x_max": 1280, "y_min": 250, "y_max": 600}
center_box = {"x_min": 415, "x_max": 1450, "y_min": 610, "y_max": 700}

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def inside_box(x, y, box):
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]

def show_mouse_coords(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Hiiren sijainti: x= {x}, y={y}")

center_box_frames = 0
fps = cap.get(cv2.CAP_PROP_FPS)

cv2.namedWindow("Futsal Tracking")
cv2.setMouseCallback("Futsal Tracking", show_mouse_coords)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    if frame_index % FRAME_SKIP != 0:
        continue

    input_frame = frame.copy()

    results = model(input_frame, conf=0.2)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names

    scale_x = frame.shape[1] / input_frame.shape[1]
    scale_y = frame.shape[0] / input_frame.shape[0]

    ball = None

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = class_names[int(cls_id)]
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        center = get_center(x1, y1, x2, y2)

        if cls_name == "person":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Player", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)

        elif cls_name == "sports ball":
            ball = center
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            ball_positions.append(center)

    # Zone logic
    if ball:
        if inside_box(ball[0], ball[1], goal_left):
            cv2.putText(frame, "GOAL LEFT!", (200, 200), font, 1.0, (0, 0, 255), 3)
        elif inside_box(ball[0], ball[1], goal_right):
            cv2.putText(frame, "GOAL RIGHT!", (950, 50), font, 1.0, (0, 0, 255), 3)
        elif inside_box(ball[0], ball[1], center_box):
            center_box_frames += 1
            center_box_time = center_box_frames / fps
            cv2.putText(frame, f"Ball in center zone: {center_box_time:.1f} s", (400, 450), font, 0.8, (255, 255, 0), 2)

    # Ball trail
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (0, 0, 255), 2)
        
    
    overlay = frame.copy()
    cv2.rectangle(overlay,
   (center_box["x_min"], center_box["y_min"]),
    (center_box["x_max"], center_box["y_max"]),
    (255, 255, 255), -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    

    cv2.imshow("Futsal Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


