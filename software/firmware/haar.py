import cv2
from optparse import OptionParser

def haarInit(cascadeXml):
    return cv2.CascadeClassifier(cascadeXml)

def haarDetectHands(classifier, im):
    return classifier.detectMultiScale(im)

if __name__ == '__main__' :
    parser = OptionParser()
    parser.add_option("-c", "--cascade-xml", dest="cascadeXml", help="Haar cascade xml file", metavar="DIR")
    parser.add_option("-i", "--image", dest="image", help="Image to work on")
    (options, args) = parser.parse_args()

    im = cv2.imread(options.image)
    hands = haarDetectHands(im, haarInit(options.cascadeXml))

    for (x,y,w,h) in hands:
        cv2.rectangle(im, (x,y), (x+w,y+h), 255)

    cv2.namedWindow("Hand Detection")
    cv2.imshow("Hand Detection", im)

    cv2.waitKey(0)

