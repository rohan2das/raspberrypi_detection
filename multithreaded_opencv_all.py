import cv2
import time
from ultralytics import YOLO
from threading import Thread, Lock

class WebcamStream:
    def __init__(self, src=0, size=(640, 480)):
        self.cap = cv2.VideoCapture(src)
        # Set resolution (optional, depends on camera support)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.frame = None
        self.stopped = False
        self.lock = Lock()

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()


# Load YOLOv8 nano model (fastest for CPU)
model = YOLO("yolov8n.pt")

# Start threaded webcam stream (0 = default webcam)
vs = WebcamStream(src=0, size=(640, 480)).start()
time.sleep(2)  # warm-up

frame_count = 0
start_time = time.time()

while True:
    frame = vs.read()
    if frame is None:
        continue

    # Run YOLO inference
    results = model(frame, imgsz=320, verbose=False)
    annotated_frame = results[0].plot()

    # FPS count
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)

    # Show annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

vs.stop()
cv2.destroyAllWindows()
