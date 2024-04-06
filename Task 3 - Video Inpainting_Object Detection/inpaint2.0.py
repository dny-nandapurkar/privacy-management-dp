import cv2
import numpy as np
import time

config_file = 'E:/ProPainter-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'E:/ProPainter-main/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'E:/ProPainter-main/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
print(classLabels)
print(len(classLabels))

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ImportError("Can't open webcam")

# Capture background for 45 seconds
start_time = time.time()
background_captured = False
background = None

while time.time() - start_time < 45:
    ret, frame = cap.read()
    if ret:
        background = frame.copy()
        cv2.imshow('Capturing Background', background)
    else:
        print("Error reading background frames")
        break

background_captured = True
cv2.destroyWindow('Capturing Background')  # Close the window after capturing the background

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    # Ensure that frame is not None
    if frame is None:
        break

    # Create a mask for inpainting
    mask = np.zeros_like(frame)

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    print(ClassIndex)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80 and classLabels[ClassInd - 1] == 'person':
                # Create a binary mask for the detected person
                mask[...] = 0
                cv2.rectangle(mask, tuple(boxes[0:2]), tuple(boxes[2:4]), (255, 255, 255), thickness=cv2.FILLED)

                # Manual blending: Replace the detected person region with the corresponding region from the background
                inpainted_frame = frame.copy()
                inpainted_frame[mask == 255] = background[mask == 255]

                # Display the inpainted frame
                cv2.imshow('Inpainted Frame', inpainted_frame)

            else:
                cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(2)
    if key == ord('q') or key == 27:  # Exit loop when 'q' or Esc key is pressed
        break

cap.release()
cv2.destroyAllWindows()
