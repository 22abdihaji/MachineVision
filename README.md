#  Machine Vision Futsal Tracking

##  Project Summary
This project performs **real-time tracking of futsal players and ball** using **YOLOv8 object detection**. Key features:

- Detects **players** and the **ball**
- Highlights if the **ball enters the center zone**
- Measures **time ball stays in that zone**
- Detects **left and right goals**
- Visualizes the **ball trail path**
- **Draws a transparent center box** on screen
- Allows user to **hover mouse** to read live coordinates for manual calibration

---

##  Tools & Technologies Used

-  **Python 3.12**
-  **YOLOv8 (Ultralytics)** – for object detection  
-  **OpenCV** – for frame-by-frame processing, drawing, and UI
-  `face_recognition` (📦 *initially used but later removed* for performance)
-  **VirtualBox** + **Kubuntu**
-  **VS Code** for code editing
-  **GitHub** for version control  
-  **Microsoft 365 Copilot** & **DeepSeek** for AI assistance

---

##  What I Have Done

- Installed & configured YOLOv8 in Linux VM
- Collected **training session videos** with different lighting and angle conditions
- Built code to:
  - Detect players, ball, and goal areas
  - Track and **draw bounding boxes**
  - Calculate how many **frames the ball stays in the center zone**
- Refactored:
  - Removed facial recognition (too slow/inaccurate)
  - Improved logic to check **bottom of ball bounding box** (more accurate for floor-level zones)
- Drew **transparent overlays** for easier visual reference
- Added **mouse coordinate feedback** (helps calibrate zones precisely)
- Created a `.gitignore` to avoid pushing large videos and model weights
- Handled YOLO model variations (e.g., `yolov8n`, `yolov8m`)
- Uploaded source code to GitHub with readme documentation

---

##  Testing Done

-  Tested on **multiple futsal video sessions**
-  Adjusted **center box coordinates** using:
  - Manual testing
  - Mouse hover logs
-  Detected ball and players at various distances
-  Measured frame-to-time accuracy using `fps`
-  Experimented with different models: `yolov8n.pt` → `yolov8m.pt`
-  Observed performance on both **slow (VirtualBox)** and **host environments**
-  Evaluated accuracy of ball zone detection (top vs. bottom of ball)

---

##  Challenges Faced

-  Ball is **white, small, and fast-moving**, difficult to detect reliably
-  Slow performance in **VirtualBox** when using heavier YOLO versions
-  Face recognition didn’t work well, removed for speed
-  **GitHub push issues** due to large `.pt` or `.mp4` files (fixed with `.gitignore`)
-  Needed exact tuning of **center zone (bounding box)** to fit floor-level ball
-  Some videos had frame rate or format issues in the Linux VM
-  Center zone was sometimes too **high or too small**, and didn’t match real floor zone

---

##  Results & Observations

-  **Player detection** is very accurate, even at range
-  **Ball trail works**, but better near the camera
-  **Ball detection less reliable** at long distance or when fast
-  **Overlay box for center zone** added — helps visualize area on the ground
-  Bounding box originally appeared too **high**; now fixed using **bottom-center point**
-  Measurement of time spent in center zone now feels realistic
-  Mouse debug feature was helpful to calibrate zone sizes

---

##  README & Code Organization

- Project now includes:
  - `README.md` with full documentation
  - `.gitignore` to skip heavy assets
  - Cleaned code with comments for every section
  - Code uploaded to GitHub
  - Final version includes all improvements and bug fixes

---

##  Next Ideas

- Train a **custom YOLO model** for futsal ball and player types
- Add **object tracking ID** to see passes between players
- Use **higher quality camera angle** to improve ball detection
- Possibly integrate OpenPose or pose estimation for player orientation

---



