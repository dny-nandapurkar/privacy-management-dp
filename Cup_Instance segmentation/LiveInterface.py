from ultralytics import YOLO
import cv2

model = YOLO("D:\DOCUMENTS\VIIT\T.Y. Sem 5\DP\cupInstance.pt") #give your own path
model.predict(source="0", show=True, conf=0.5)