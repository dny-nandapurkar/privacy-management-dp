import cv2
import numpy as np

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

#low_orange = np.array([5, 50, 50])
#high_orange = np.array([15, 255, 255])
cap =cv2.VideoCapture(0)


while True:
  ret,frame = cap.read()
  cv2.imshow('Original Frame', frame)
  
  # Convert to HSV image
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  #cv2.imshow('HSV', hsv)
  
  # Mask Image
  mask = cv2.inRange(hsv, low_green, high_green)
  cv2.imshow('Masked Frame', mask)

  # Find contours (Sape outlines)
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(frame, contours, -1,(0,0,255),2)
  cv2.imshow('Frame', frame)

  # highlight only required size masked shapes
  for c in contours:
    area = cv2.contourArea(c)
    # remove noise small sections
    if area > 400:
      x, y, w, h = cv2.boundingRect(c)
      cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0),2)
      cv2.drawContours(frame, c, -1, (0, 0 ,255),2)
    print(area)
    cv2.imshow('Frame',frame)

  if cv2.waitKey(1) == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()