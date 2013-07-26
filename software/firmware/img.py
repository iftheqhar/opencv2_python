import cv2
ramp_frames = 100
camera = cv2.VideoCapture(0)
def get_image():
    retval, im = camera.read()
    return im
for i in xrange(ramp_frames):
 temp = get_image()
camera_capture = get_image()
file = "C:\test_image.png"
print("Taking image...")
cv2.imwrite(file, camera_capture)
cv2.imshow("lll",temp)
