import os
import cv2
import numpy as np
import face_recognition
from collections import deque, Counter
from ultralytics import YOLO

# Parameters
FRAME_SKIP = 2  # Skip every N frames for speed
YOLO_SIZE = 640  # Resize width for YOLO
FACE_MATCH_THRESHOLD = 0.6

# Load face encodings
face_dir = "face_dataset"
known_face_encodings = []
known_face_names = []

for person in os.listdir(face_dir):
    person_folder = os.path.join(face_dir, person)
    for img_file in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(person)

# Load YOLOv8
model = YOLO("yolov8n.pt")  # Use yolov8n for speed
model.fuse()

# Load video
video_path = "futsal vid.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening the video file")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(f"FPS: {fps} | Total frames: {frame_count}")

# Tracking
font = cv2.FONT_HERSHEY_SIMPLEX
pass_log = []
current_holder = None
ball_positions = deque(maxlen=50)
frame_index = 0
central_box = {
    "x_min": 200,
    "y_min": 300,
    "x_max": 1050,
    "y_max": 600
}
def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def inside_central_box(x, y, box):
    return box["x_min"] <= x <= box["x_max"] and box["y_min"] <= y <= box["y_max"]



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_index += 1
    if frame_index % FRAME_SKIP != 0:
        continue

    # Resize for faster YOLO inference
    input_frame = cv2.resize(frame, (YOLO_SIZE, int(frame.shape[0] * YOLO_SIZE / frame.shape[1])))

    # Run YOLO
    results = model(input_frame, conf=0.4)
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names

    scale_x = frame.shape[1] / input_frame.shape[1]
    scale_y = frame.shape[0] / input_frame.shape[0]

    players = []
    ball = None

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cls_name = class_names[int(cls_id)]

        # Scale back to original
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if cls_name == "person":
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (150, 150))
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            name = "Player"
            if encodings:
                distances = face_recognition.face_distance(known_face_encodings, encodings[0])
                if len(distances) > 0 and distances.min() < FACE_MATCH_THRESHOLD:
                    idx = np.argmin(distances)
                    name = known_face_names[idx]

            center = get_center(x1, y1, x2, y2)
            if not inside_central_box(center[0], center[1], central_box):
                continue
            players.append((name, center))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)

        elif cls_name == "sports ball":
            ball = get_center(x1, y1, x2, y2)
            cv2.circle(frame, ball, 5, (0, 0, 255), -1)
            ball_positions.append(ball)

    if ball:
        distances = [(name, euclidean(ball, center)) for name, center in players if name != "Unknown"]
        if distances:
            name, dist = min(distances, key=lambda x: x[1])
            if dist < 100:
                if current_holder and current_holder != name:
                    pass_log.append((current_holder, name))
                current_holder = name

    for i in range(1, len(ball_positions)):
        cv2.line(frame, ball_positions[i - 1], ball_positions[i], (0, 0, 255), 2)

    cv2.imshow("Fast Futsal Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Passing summary
counter = Counter([p[0] for p in pass_log])
total = sum(counter.values())
print("\nPassing Stats:")
for player, count in counter.items():
    acc = count / total * 100 if total else 0
    print(f"{player}: {count} passes ({acc:.1f}%)")
