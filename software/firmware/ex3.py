import cv2
import numpy as np
img=cv2.imread('C:\Python27\hand1.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
ret,thresh1 = cv2.threshold(blur,70,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    #thresh1 = cv2.adaptiveThreshold(thresh1,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            #cv2.THRESH_BINARY,11,2)
contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
drawing = np.zeros(img.shape,np.uint8)

    
    #print(len(contours))
    #cnt=contours[0]
    
for cnt in contours:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(drawing,[cnt],0,(0,255,0),2) # draw contours in green color
            cv2.drawContours(drawing,[hull],0,(0,0,255),2) # draw contours in red color
            
cv2.imshow('output',drawing)
cv2.imshow('input',img)
