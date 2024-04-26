import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pickle

import matplotlib
matplotlib.use('TkAgg')

img = cv2.imread('truck.jpg')
model = YOLO('yolov8n.pt')
objects = model(img, save = True, classes = [7])   # Change class index here

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Define class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocolli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

# Initialize an empty mask for overlaying the segmentation masks
overlay_mask = np.zeros_like(img[:, :, 0])

# Process each detection
for result in objects:
    boxes = result.boxes
    cls = boxes.cls

    for i in range(len(boxes)):
        class_index = int(cls[i])
        class_name = class_names[class_index]

        if class_index == 7:   # Change class index here
            # Get the coordinates of bounding box
            x1, y1, x2, y2 = boxes.xyxy[i]

            # Plot rectangle around the detected object
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Add text to the rectangle
            text = class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = int(x1 + 5)
            text_y = int(y1 + text_size[1] + 5)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)

            # Perform segmentation
            import sys
            sys.path.append("..")
            from segment_anything import sam_model_registry, SamPredictor

            sam_checkpoint = "sam_vit_h_4b8939.pth"
            model_type = "vit_h"

            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

            predictor = SamPredictor(sam)
            predictor.set_image(img)

            input_box = np.array([x1, y1, x2, y2])

            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )

            # Accumulate the segmentation mask
            overlay_mask += masks[0]

# Normalize the overlay mask
overlay_mask = (overlay_mask > 0).astype(np.uint8) * 255

# Apply overlay mask on the original image
overlayed_img = cv2.addWeighted(img, 0.7, cv2.merge([np.zeros_like(overlay_mask), overlay_mask, np.zeros_like(overlay_mask)]), 0.3, 0)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(overlayed_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('output.png')
plt.show()