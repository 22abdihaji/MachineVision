# 🎯 Machine Vision Futsal Tracking

## 📌 Project Summary
This project tracks players and the ball from futsal match videos using YOLOv8 object detection. It:
- Detects players and the ball
- Tracks if the ball enters a center zone
- Measures how long the ball stays in that area
- Detects left and right goals
- Visualizes the ball path

## 🧰 Tools Used
- **Python 3.12**
- **YOLOv8 (Ultralytics)**
- **OpenCV**
- **face_recognition** (previously)
- **VirtualBox + Kubuntu**
- **GitHub + VS Code**

## 🧪 Testing Done
- Tested player tracking on multiple sessions
- Calibrated center zone for accurate detection
- Attempted face recognition but dropped due to accuracy
- Resolved virtual environment and dependency issues

## ⚠️ Challenges
- Detecting small, fast-moving white ball
- Video format issues in Linux VM
- GitHub push errors due to video/model size
- Model performance vs frame rate balance

## ✅ Results
- Accurate player box detection
- Ball trails working
- Goals detected
- Center zone time tracked

## 🗂️ Repo Structure
