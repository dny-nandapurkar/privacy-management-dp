import cv2
import matplotlib.pyplot as plt
import sqlite3
import datetime

# Database functions
def initialize_db():
    conn = sqlite3.connect('videos.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos
        (id INTEGER PRIMARY KEY, video_path TEXT, datetime TEXT)
    ''')
    conn.commit()
    conn.close()

def save_video_path(path):
    conn = sqlite3.connect('videos.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO videos (video_path, datetime) VALUES (?, ?)", (path, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()

# Initialize database
initialize_db()

# Load model and labels
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
with open('labels.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Video capture setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ImportError("Can't open webcam")

# Video writing setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'output.avi'
out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    if not ret:
        break

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    if len(ClassIndex) != 0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd-1], (boxes[0]+10, boxes[1]+40), font, fontScale=font_scale, color=(0, 255, 0), thickness=3)

    out.write(frame)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

# Cleanup and save path
cap.release()
out.release()
cv2.destroyAllWindows()
save_video_path(output_path)
