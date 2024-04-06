import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

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

start_time = time.time()
background = None

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

while True:
    ret, frame = cap.read()

    # Ensure that frame is not None
    if frame is None:
        break

    if time.time() - start_time < 45:  # Capture background for 45 seconds
        ret, frame = cap.read()
        if ret:
            background = frame.copy()  # Store background frame
        else:
            print("Error reading background frames")
            break
    else:
        break  # Exit background capture loop


    # Create a mask for inpainting
    mask = np.zeros_like(frame)

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold = 0.55)

    print(ClassIndex)
    if len(ClassIndex)!=0:
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            if (ClassInd <= 80) and classLabels[ClassInd - 1] == 'person':
                # Create a binary mask for the detected person
                mask[...] = 0
                cv2.rectangle(mask, tuple(boxes[0:2]), tuple(boxes[2:4]), (255, 255, 255), thickness=cv2.FILLED)

                # Inpaint the detected person
                #mask_smoothed = cv2.GaussianBlur(mask, (15, 15), 0)
                if background is not None:
                    inpainted_frame = cv2.inpaint(frame, mask[..., 0], background=background) #inpaintRadius=10, flags=cv2.INPAINT_NS)
                else:
                    print("Background not captured or unavailable")
                    # Use existing inpainting method as a fallback
                    inpainted_frame = cv2.xphoto.createSimpleWB()
                    inpainted_frame.inpaint(frame, mask[..., 0])

                # Exemplar based inpainting
                #inpainted_frame = cv2.xphoto.createSimpleWB()
                #inpainted_frame.inpaint(frame, mask[..., 0])


                # Display the inpainted frame
                cv2.imshow('Inpainted Frame', inpainted_frame)
                #cv2.rectangle(frame,boxes,(255,0,0),2)
                #cv2.putText(frame, classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale = font_scale, color=(0, 255, 0),thickness=3)

            else:
                cv2.imshow('Object Detection', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()