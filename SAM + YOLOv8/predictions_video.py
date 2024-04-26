import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# Load YOLO model
model = YOLO('yolov8n.pt')

# Define class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocolli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    objects = model(frame, save=False, classes=[0])  # Change class index here

    # Initialize an empty mask for the current frame
    overlay_mask = np.zeros_like(frame[:, :, 0])

    # Process each detection
    for result in objects:
        boxes = result.boxes
        cls = boxes.cls

        for i in range(len(boxes)):
            class_index = int(cls[i])

            if class_index == 0:  # Change class index here
                # Get the coordinates of bounding box
                x1, y1, x2, y2 = boxes.xyxy[i]

                # Perform segmentation
                sam_checkpoint = "sam_vit_h_4b8939.pth"
                model_type = "vit_h"

                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

                predictor = SamPredictor(sam)
                predictor.set_image(frame)

                input_box = np.array([x1, y1, x2, y2])

                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )

                # Accumulate the segmentation mask
                overlay_mask += masks[0]

                # Plot rectangle around the detected object
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # Add text to the rectangle
                text = class_names[class_index]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 4
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = int(x1 + 5)
                text_y = int(y1 + text_size[1] + 5)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

    # Normalize the overlay mask
    overlay_mask = (overlay_mask > 0).astype(np.uint8) * 255

    # Apply overlay mask on the original frame
    overlayed_frame = cv2.addWeighted(frame, 0.7, cv2.merge([np.zeros_like(overlay_mask), overlay_mask, np.zeros_like(overlay_mask)]), 0.3, 0)

    # Display the resulting frame
    cv2.imshow('Real-time Object Segmentation', overlayed_frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()