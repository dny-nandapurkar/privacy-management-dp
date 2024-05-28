import cv2
import matplotlib.pyplot as plt

config_file = 'E:/ProPainter-main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'E:/ProPainter-main/frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'E:/ProPainter-main/labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read())
print(classLabels)
print(len(classLabels))

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


img = cv2.imread('E:\ProPainter-main\Image.jpg')
#plt.imshow(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
print(ClassIndex)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
    #cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    #cv2.putText(im, text, (text_offset_x, test_offset_y), font, fontScale=font_scale, color=(0,0,0), thickness=1)
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img, classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale = font_scale, color=(0, 255, 0),thickness=3)
    
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# VIDEO_DEMO
    
cap = cv2.VideoCapture(1)

# Check if the video is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise ImportError("Can't open webcam")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)

    print(ClassIndex)
    if (len(ClassIndex!=0)):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <= 80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame, classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale = font_scale, color=(0, 255, 0),thickness=3)
    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()