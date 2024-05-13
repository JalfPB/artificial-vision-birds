import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
#model_species = WIP

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

def detect_bird(frame):
    """Detects birds in a frame."""
    results = bird_detector.predict(source=frame, show=False) 
    for result in results:
        boxes = result.boxes.data 
        classes = result.boxes.cls 
        confidences = result.boxes.conf 
        if (classes == 14).any() and (confidences > 0.5).any():
            return True, boxes[0]
    return False, None

def predict_race(frame, bird_bbox):
    """Predicts the species of the bird in the given frame.

    Args:
        frame: The frame containing the bird.
        bird_bbox: Bounding box coordinates of the bird (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = map(int, bird_bbox)  # Convert to integers
    bird_crop = frame[y1:y2, x1:x2]  # Crop the bird from the frame

    #race predictor its the name of the model WIP
    species = species_predictor.predict(bird_crop)

    return species



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error reading frame from camera")
        break 

    processed_frame = process_frame(frame)

    print("Frame shape:", processed_frame.shape)

    cv2.imshow("Webcam", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()