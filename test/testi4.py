import cv2
import numpy as np
from ultralytics import YOLO  


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture("/home/centria/projects/machinevision/IMG_9397.mp4")

if not cap.isOpened():
    print("Cannot open video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    
    results = model(frame, conf=0.25)

    
    detections = results[0].boxes.data.cpu().numpy()
    class_names = results[0].names

    for det in detections:
        xmin, ymin, xmax, ymax, conf, cls_id = det
        label = class_names[int(cls_id)]

        
        if label in ['person', 'sports ball']:
            color = (0, 255, 0) if label == 'person' else (0, 0, 255)
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (255, 255, 0), 2)  # Cyan for field

    
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    
    contours, _ = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h != 0 else 0

        
        if 100 < area < 3000 and (aspect_ratio > 2 or aspect_ratio < 0.5):
            cv2.drawContours(frame, [cnt], -1, (255, 0, 255), 2)  # Purple for court lines

    
    cv2.imshow("YOLOv8 Futsal Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
