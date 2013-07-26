import sys
import cv2.cv as cv
storage=cv.CreateMemStorage(0)
image _scale =1.3
haar _scale =1.2
min _neighbors =1
haar _flags =0
def detect_and_draw(img):
    # allocate temporary images
    gray=cv.CreateImage((img.width,img.height),8,1)s m a l l _ i m g = c v . C r e a t e I m a g e ( ( cv.Round(img.width/image _scale),
    cv.Round(img.height/image _ scale)), 8, 1 )
    # convert color input image to grayscale
    cv.CvtColor( img, gray, cv.CV _ BGR2GRAY )
    # scale input image for faster processing
    cv.Resize( gray, small _ img, cv.CV _ INTER _ NN )
    cv.EqualizeHist( small _ img, small _ img )
    # start detection
    if( cascade ):
       faces=cv.HaarDetectObjects( small _ img,  cascade, storage,haar _ scale, min _ neighbors, haar _ flags )
if faces:
    for(x,y,w,h),n in faces:
        pt1=(int(x*image _ scale),int(y*image _ scale))
        pt2=(int((x+w)*image _ scale),
        int((y+h)*image _ scale))
         
        cv.Rectangle(img,pt1,pt2,cv.CV _ RGB(255,0,0),3,8,0)
        cv.ShowImage( “result”, img )
if__name__==‘__main__’:
   cascade _ name=“./haarcascade _ frontalfacealt _ tree.xml”
   cascade=cv.Load(cascade _ name)
   # Start capturing.Can change index if more than one
   camera present
   capture=cv.CaptureFromCAM(0)
   # Create the output window
   cv.NamedWindow(“result”,1)
   frame _ copy=None
while True:
    frame=cv.QueryFrame( capture )
    # make a copy of the captured frame
    if not frame _ copy:
        frame _ copy=cv.CreateImage((frame.width,fram e.height),cv.IPL _ DEPTH _ 8U, frame.nChannels )
        cv.Copy( frame, frame _ copy )
    detect _ and _ draw(frame _ copy)
    c=cv.WaitKey(7)
    if
    c==27:
    # Escape pressed
    break
