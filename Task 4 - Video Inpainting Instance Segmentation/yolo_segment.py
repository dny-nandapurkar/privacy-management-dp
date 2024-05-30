import cv2
import numpy as np
import time
from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')      # or use 'yolov8n-seg.pt'

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ImportError("Can't open webcam")

print("Capturing background frame for 45 seconds. Please ensure the frame is clear.")
start_time = time.time()
background = None
while time.time() - start_time < 45:
    ret, frame = cap.read()
    if ret:
        background = frame.copy()
        cv2.imshow('Capturing Background', background)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error reading background frames")
        break
print("Background frame captured.")
cv2.destroyWindow('Capturing Background')

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    results = model(frame)[0]

    # Check if masks are available
    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Create a mask for inpainting
        mask = np.zeros_like(frame[:, :, 0])

        for i in range(len(boxes)):
            if classes[i] == 0 and scores[i] > 0.5:
                mask[masks[i] > 0.5] = 255
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Inpaint the frame using the background
        inpainted_frame = frame.copy()
        inpainted_frame[mask == 255] = background[mask == 255]

        cv2.imshow('Inpainted Frame', inpainted_frame)
    else:
        cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(2)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()