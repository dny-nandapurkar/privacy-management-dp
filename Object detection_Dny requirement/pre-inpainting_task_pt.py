import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("Pre-inpainting task", annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()