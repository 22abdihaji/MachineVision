import os
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO  # YOLOv8 object detection model

# Parameters
FRAME_SKIP = 2  # Process every 2nd frame to reduce computation
YOLO_SIZE = 640  # YOLO input image size

# Load YOLOv8 model
model = YOLO("yolov8m.pt")  # Load YOLOv8 medium model
model.fuse()  # Optimize model (for slightly faster inference)

# Open video
cap = cv2.VideoCapture("futsal vid.mp4")  # Load futsal video
if not cap.isOpened():
    print("Error opening video")
    exit()

# Drawing and tracking setup
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for labels
ball_positions = deque(maxlen=50)  # Store previous ball positions (trail)
frame_index = 0  # To handle frame skipping

# Define detection zones (as rectangles)
goal_left = {"x_min": 100, "x_max": 150, "y_min": 250, "y_max": 600}
goal_right = {"x_min": 1100, "x_max": 1280, "y_min": 250, "y_max": 600}
center_box = {"x_min": 415, "x_max": 1450, "y_min": 610, "y_max": 710}  # zone for ball control

# Utility function to find center of a bounding box
def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

# Utility function to check if a point is inside a zone
def inside_box(x, y, box):
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]

# Debug function: prints mouse x/y position over video window
def show_mouse_coords(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Hiiren sijainti: x= {x}, y={y}")  # For manual zone calibration

# Zone timing variables
center_box_frames = 0  # How many frames ball has been in center
fps = cap.get(cv2.CAP_PROP_FPS)  # Read video FPS

# Setup mouse tracking
cv2.namedWindow("Futsal Tracking")
cv2.setMouseCallback("Futsal Tracking", show_mouse_coords)

# Frame-by-frame video loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    if frame_index % FRAME_SKIP != 0:
        continue

    input_frame = frame.copy()  # Copy for processing

    results = model(input_frame, conf=0.2)  # Run YOLO model
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names

    scale_x = frame.shape[1] / input_frame.shape[1]
    scale_y = frame.shape[0] / input_frame.shape[0]

    ball = None  # Reset ball per frame

    # Loop through detections
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = class_names[int(cls_id)]
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        center = get_center(x1, y1, x2, y2)

        if cls_name == "person":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for player
            cv2.putText(frame, "Player", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)

        elif cls_name == "sports ball":
            ball = center  # Save ball center
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot for ball
            ball_positions.append(center)

    # Zone logic
    if ball:
        if inside_box(ball[0], ball[1], goal_left):
            cv2.putText(frame, "GOAL LEFT!", (200, 200), font, 1.0, (0, 0, 255), 3)
        elif inside_box(ball[0], ball[1], goal_right):
            cv2.putText(frame, "GOAL RIGHT!", (950, 50), font, 1.0, (0, 0, 255), 3)

    # Improve ball zone accuracy: check bottom-center of bounding box
    ball_rect_bottom_center = ((x1 + x2) // 2, y2)
    if inside_box(ball_rect_bottom_center[0], ball_rect_bottom_center[1], center_box):
        center_box_frames += 1
        center_box_time = center_box_frames / fps
        cv2.putText(frame, f"Ball in center zone: {center_box_time:.1f} s",
                    (400, 450), font, 0.8, (255, 255, 0), 2)

    # Draw trail behind ball
    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (0, 0, 255), 2)

    # Draw translucent center box
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (center_box["x_min"], center_box["y_min"]),
                  (center_box["x_max"], center_box["y_max"]),
                  (255, 255, 255), -1)  # filled white box
    alpha = 0.2  # transparency level
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # merge overlay with frame

    # Show processed frame
    cv2.imshow("Futsal Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q' key
        break

cap.release()  # Release the video file
cv2.destroyAllWindows()  # Close all OpenCV windows
