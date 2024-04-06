import cv2
import tensorflow as tf
import numpy as np

# Load the TensorFlow model
model = tf.saved_model.load('yolov8n-seg.pb\saved_model.pb')
infer = model.signatures['serving_default']

# Start capturing from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error opening webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB, resize, convert to float32, and add batch dimension
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))  # Adjust size as needed
    input_tensor = tf.convert_to_tensor([frame_resized], dtype=tf.float32)

    # Run inference
    output = infer(images=input_tensor)

    # Post-process and visualize the results here
    # ...

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()