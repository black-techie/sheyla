import torch
import cv2
import numpy as np
from torchvision import transforms

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x

# Function to detect phones in an image
def detect_phone(image):
    results = model(image)
    # Extract bounding boxes and labels
    bbox = results.xyxy[0].numpy()  # Bounding boxes
    labels = results.names           # Class names
    phones = []

    for box in bbox:
        if labels[int(box[5])] == 'cell phone':
            phones.append(box[:4])  # Extract bounding box coordinates

    return phones

# Load video
cap = cv2.VideoCapture('path_to_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect phones
    phones = detect_phone(rgb_frame)

    # Draw bounding boxes
    for phone in phones:
        x1, y1, x2, y2 = map(int, phone)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
