#  Machine Vision Futsal Tracking

##  Project Summary
This project tracks players and the ball from futsal training videos using YOLOv8 object detection. It:
- Detects players and the ball
- Tracks if the ball enters a center zone
- Measures how long the ball stays in that area
- Detects left and right goals
- Visualizes the ball path

##  Tools & Technologies Used
- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **face_recognition** (initially used, later removed for speed)
- **VirtualBox + Kubuntu**
- **GitHub + VS Code**
- **Microsoft 365 Copilot & DeepSeek**


## What I have Done
- Set up YOLOv8 inside VirtualBox
- Collected futsal training video samples.
- Detected:
- Players
- Ball
- Goals
- Time the ball stays in the central area
- Refactored code to improve performance (removed face recognition).
- Tracked and annotated players using bounding boxes.
- Created bounding boxes for goal zones and central box.
- Measured frame-based time calculations for zone logic.
- Took some pictures from the players and screenshots so that identification works properly.
  

##  Testing Done
- Tested player tracking on multiple sessions
- Calibrated center zone for accurate detection
- Attempted face recognition but dropped due to accuracy
- Resolved virtual environment and dependency issues
- Tried to measure ball sizes so that it detects on the video.
- Tested all the videos that I took from the futsal trainings.
- Changed central box sizes multiple times so it detects.
- 

##  Challenges
- Detecting small, fast-moving white ball
- Video format issues in Linux VM
- GitHub push errors due to video/model size
- Model performance vs frame rate balance
- Detecting the goals, players and frames on the field.
- Difficult to get the video faster with YOLOv5n and YOLOv8n.

##  Results
- Accurate player box detection
- Ball trails working, but only close to the camera and sometimes.
- Makes lines sometimes, which means that with better camera and camera angle possible to detect the ball. 
- Center zone identified, but makes the box in different location and smaller than I expected.


