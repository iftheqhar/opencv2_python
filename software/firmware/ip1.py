import cv2
cv2.namedWindow("lll")
cap = cv2.VideoCapture(0)
x=100
if(cap.isOpened()):
   ret,img2 = cap.read()
   ret,img2 = cap.read()
   ret,img2 = cap.read()
   ret,img2 = cap.read()
   ret,img2 = cap.read()
   img2=img2[:,:,1]
   #img2[img2[:,:]>x]=255
   #img2[img2[:,:]<x]=0
   ret,img2 = cv2.threshold(img2,150,255,cv2.THRESH_BINARY_INV)
#img2 = cv2.imread('C:\Users\KAMAL\Pictures\pic.jpg')

while( cap.isOpened() ) :
    ret,img = cap.read()
    img=img[:,:,1]
    ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY_INV)
    #img[img[:,:]>x]=255
    #img[img[:,:]<x]=0
    dst=cv2.absdiff(img, img2)
   
    
    #img=img[:,:,1]
    #image=img[:,:]-img2[:,:]
    #image[image[:,:]==255]=0
    #image[image[:,:]==0]=0 
    #image[~(image[:,:]==0)]=img[~(image[:,:]==0)]
    #image[image[:,:]<128]=255
    #image[image[:,:]>128]=0
    #image[image[:,:]!=0]=0
    
    cv2.imshow("lll",img)
    k = cv2.waitKey(10)
    if k == 27: 
        break
