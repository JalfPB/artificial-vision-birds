import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

def process_frame(frame):
    results = model.predict(source=frame, show=False)  
    
    for result in results:
        boxes = result.boxes.data  
        classes = result.boxes.cls  
        confidences = result.boxes.conf  
        
        if (classes == 0).any() and (confidences > 0.5).any():
            print("Person detected!")
            break  
        
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame from camera")
        break 

    processed_frame = process_frame(frame)

    print("Frame shape:", processed_frame.shape)  # Verify frame dimensions

    cv2.imshow("Webcam", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()