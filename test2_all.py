import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import Transform
from threading import Thread, Lock

class PiCameraStream:
    def __init__(self, size=(640, 480)):
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_video_configuration(main={"size": size},
            transform=Transform(hflip=True, vflip=True))
        )
        self.picam2.start()
        self.frame = None
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            frame = self.picam2.capture_array()
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.picam2.stop()


# ✅ Load YOLOv8 nano model (fastest for CPU)
model = YOLO("yolov8n.pt")

# Start threaded PiCamera stream
vs = PiCameraStream(size=(1536, 864)).start()
time.sleep(2)  # warm-up

frame_count = 0
start_time = time.time()

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Convert RGBA → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    results = model(frame, imgsz=320, verbose=False)

    annotated_frame = results[0].plot()

    # FPS count
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)
    
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    # Show annotated frame
    cv2.imshow("YOLOv8 Detection", rgb_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

vs.stop()
cv2.destroyAllWindows()
