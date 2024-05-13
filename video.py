import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8s.pt")

def process_frame(frame):
    results = model.predict(source=frame, show=False) 
    for result in results:
        boxes = result.boxes.data 
        classes = result.boxes.cls 
        confidences = result.boxes.conf 
        if (classes == 14).any() and (confidences > 0.5).any():
            print("Pájaro detectado!")
            break 
    return frame

cap = cv2.VideoCapture('./Videos/birds.mp4') 

if not cap.isOpened():
    print("Error opening video file")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Reached end of video")
        break 

    processed_frame = process_frame(frame)
    print("Frame shape:", processed_frame.shape)  # Verify frame dimensions
    cv2.imshow("Video", processed_frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()