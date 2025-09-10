# Raspberry Pi Detection

This repository contains two simple Python scripts for running **real-time object detection** on Raspberry Pi or PC using [YOLOv8](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/).  

- `basic_picamera2_all.py` → Uses **Raspberry Pi Camera (Picamera2)**.  
- `multithreaded_opencv_all.py` → Uses **USB/Webcam** with a **multithreaded video stream** for better performance.  

Both scripts display live detections with bounding boxes and show FPS in real-time.  

---

## Features
- Works with **Raspberry Pi Camera** (libcamera / Picamera2) or **USB Webcam**  
- Real-time YOLOv8 inference on CPU (optimized with YOLOv8-nano)  
- FPS counter overlay for performance monitoring  
- Multithreaded video capture for smoother results (OpenCV version)  
- Simple, minimal, and easy to extend  

---

## Files
```
raspberrypi_detection/
├── basic_picamera2_all.py       # Detection using Raspberry Pi Camera
├── multithreaded_opencv_all.py  # Detection using USB/PC Webcam
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/rohan2das/raspberrypi_detection.git
cd raspberrypi_detection
```

### 2. Create & activate virtual environment
```bash
sudo apt update
sudo apt full-upgrade
sudo apt install -y python3-picamera2
python3 -m venv venv_detection --system-site-packages 
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Run with Raspberry Pi Camera
```bash
python basic_picamera2_all.py
```

### Run with USB/PC Webcam
```bash
python multithreaded_opencv_all.py
```

Press `ESC` to exit the detection window.  

---

## 🛠️ Requirements
- Raspberry Pi 4 / 5 (for PiCamera script)  
- Raspberry Pi Camera Module (libcamera + Picamera2)  
- OR USB Webcam (for OpenCV script)  
- Python 3.9+  
- Dependencies listed in `requirements.txt` 