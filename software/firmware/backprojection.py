import cv2
import numpy as np
from matplotlib import pyplot as plt
 
#roi is the object or region of object we need to find
roi = cv2.imread('hand.jpg')
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
cap = cv2.VideoCapture(0)
#target is the image we search in
while( cap.isOpened() ) :
    ret,target = cap.read()
    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
     
    # Find the histograms. I used calcHist. It can be done with np.histogram2d also
    
    I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    R = M/(I+1)
    h,s,v = cv2.split(hsvt)
    B = R[h.ravel(),s.ravel()]
    B = np.minimum(B,1)
    B = B.reshape(hsvt.shape[:2])
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(B,-1,disc,B)
    B = np.uint8(B)
    cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
    ret,thresh = cv2.threshold(B,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    #res = cv2.bitwise_and(target,thresh) 
    #res = np.vstack((target,thresh,res))
    cv2.imshow('res.jpg',thresh)
