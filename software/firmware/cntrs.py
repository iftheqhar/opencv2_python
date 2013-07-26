import cv2
import numpy as np
template = cv2.imread('hand.jpg')
cap = cv2.VideoCapture(0)
while( cap.isOpened() ) :
    ret,img = cap.read()
    fgmask=cv2.BackgroundSubtractor.apply(img)
    cv2.imshow('output',fgmask)
    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()
