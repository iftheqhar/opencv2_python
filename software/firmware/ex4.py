import cv
import cv2
import numpy as np
import copy
from optparse import OptionParser
import haar
import os


def loadRawSample(im):
  im = np.asarray(im)
  im = 255-(im/np.max(im)*255).astype('uint8')
  return im

def loadSample(im):

  img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('uint8')
  return img

def extractBinary(img):
  element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
  #img = cv2.equalizeHist(img)

  # remove artefacts and noise
  img = cv2.erode(img, element)
  #img = cv2.dilate(img, element)

  # renormalize
  #img = ((img/np.max(img).astype('float'))*255).astype('uint8')

  thresh = findThresh(smoothHist(img))

  if thresh is not None:
    _, imb = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
  else:
    _, imb = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return imb

def drawPolygon(im, points, color, thickness=1):

  first = None
  last = None
  prev = None

  for p in points:
    if first == None:
      first = p
    else:
      cv2.line(im, prev, p, color, thickness)

    prev = p
    last = p

  cv2.line(im, last, first, color, thickness)

def drawPoints(im, points, color, radius = 2):
  for p in points:
    cv2.circle(im, p, radius, color, -1)

def drawOrientation(im, ellipse, color, thickness):
  e = ellipse
  cv2.ellipse(im, (e[0], (0, e[1][1]), e[2]), color, thickness)

def bestContourAsInt(contours, minArea = -1):
  maxArea = -1
  contour = None

  for cnt in contours:
    cnt_int = cnt.astype('int')
    area = cv2.contourArea(cnt_int)
    if(area > maxArea and area > minArea):
      contour = cnt_int
      maxArea = area

  return contour

def refineHullDefects(hull, defects, contour, thresh):
  hull_refined = list(hull)
  defects_points = list()

  for d in defects:
    index = hull.index(tuple(contour[d[0][0]][0]))
    value = tuple(contour[d[0][2]][0])
    
    if(d[0][3] > thresh):
      hull_refined.insert(index, value)
      defects_points.append(value)

  return hull_refined, defects_points

def drawResult(im, features):
  imc = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

  drawPolygon(imc, features.get('hull'), (0, 255, 255), 2)
  drawPolygon(imc, features.get('shape'), (0, 255, 0), 2)
  drawPoints(imc, features.get('defects'), (255, 0, 0), 4)
  drawPoints(imc, [features.get('centroid')], (255, 0, 255), 6)
  drawOrientation(imc, features.get('boundingellipse'), (0, 0, 255), 1)
  
  return imc

def packFeatures(contour, hull, defects, shape, rect):
  ellipse = cv2.fitEllipse(contour)

  #(x,y,w,h) = rect if rect is not None else (0,0,shape[1],shape[0])

  M = cv2.moments(contour)
  centroid_x = int(M['m10']/M['m00'])
  centroid_y = int(M['m01']/M['m00'])
  center = (centroid_x, centroid_y)


  return {'contour': contour, 'hull': hull, 'defects_nb': len(defects), 'defects': defects, 'shape': shape, 'boundingellipse': ellipse, 'angle': ellipse[2], 'centroid': center, 'rect': rect}

def findROI(img, haarc):
  hands = haar.haarDetectHands(haarc, img)

  maxi = 0
  rect = None
  for (x,y,w,h) in hands:
    if(w*h > maxi):
      maxi = w*h
      rect = (x,y,w,h)

  if rect is not None:
    (x,y,w,h) = rect
    imr = img[y:y+h, x:x+w]
  else:
    imr = img

  cv2.namedWindow("ROI")
  cv2.imshow("ROI", imr)
  
  return imr, rect

cap = cv2.VideoCapture(0)
while( cap.isOpened() ) :  
  ret,img = cap.read()
  return process(loadSample(im), haarc)

def process(im, haarc=None,silent=False):
  
  if haarc is None:
    haarc = haar.haarInit(im + '/../haar/cascade.xml')

  img_ref = im
  img, rect = findROI(img_ref, haarc)
  imb = extractBinary(img)
  imb_contours = imb.copy()
  vect = None
  img_tr = np.copy(img_ref)
  
  if not silent:
    debugThresh(img)

  if rect is None:
    img_ref = cv2.cvtColor(img_ref,cv2.COLOR_GRAY2BGR)
    return img_ref, img_ref, None

  contours, _ = cv2.findContours(imb_contours, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  if contours:
    contour = bestContourAsInt(contours)
    hull = cv2.convexHull(contour, returnPoints=False).astype('int')
    defects = cv2.convexityDefects(contour, hull)

    hull_points = [tuple(p[0]) for p in cv2.convexHull(contour, returnPoints=True)]
    contour_points = [tuple(p[0]) for p in contour]

    hull_refined, defects_points = refineHullDefects(hull_points, defects, contour, 2500)

    features = packFeatures(contour, hull_points, defects_points, hull_refined, rect)

    img = drawResult(img, features)

    img_ref = cv2.cvtColor(img_ref,cv2.COLOR_GRAY2BGR)
        
    (x,y,w,h) = rect
    img_ref[y:y+h, x:x+w] = img
    cv2.rectangle(img_ref, (x,y), (x+w,y+h), (255,0,0))
    img_tr[y:y+h, x:x+w] = imb
  

  else:
    img_ref = cv2.cvtColor(img_ref,cv2.COLOR_GRAY2BGR)
    img_tr = imb

  densityVect = zoning(imb)
  img_tr = cv2.cvtColor(img_tr,cv2.COLOR_GRAY2BGR)
  
  return img_ref, img_tr,densityVect

def smoothHist(im):
  hist_item = cv2.calcHist([im],[0],None,[256],[0,255])
  cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
  hist=np.int32(np.around(hist_item))

  data = [int(x[0]) for x in hist]
  b = 1/5.
  data = np.convolve([b,b,b,b,b], data, 'same').astype('uint8')
  data = np.convolve([b,b,b,b,b], data, 'same').astype('uint8')

  return data

def findThresh(a):
  peaks = []
  last = -999
  default = None

  if len(a) < 3:
    return default

  for i in range(3, len(a)-3):
    if(a[i] > a[i-3] and a[i] > a[i+3] and i-last > 4 ):
      last = i
      peaks.append(i)

  if len(peaks) > 1:
    return int(peaks[-2] + (peaks[-1] - peaks[-2])/2)
  else:
    return default

def debugThresh(im):
  cv2.namedWindow("dt")
  h = np.zeros((300,256,3))
  bins = np.arange(256).reshape(256,1)
  color = (255,0,0)

  data = smoothHist(im)

  pts = np.column_stack((bins,data))
  cv2.polylines(h,[pts],False,color)
  h=np.flipud(h).astype('uint8')

  thresh = findThresh(data)

  if thresh is not None:
    #print thresh, data[thresh]
    drawPoints(h, [(int(thresh), int(300-data[thresh]))], (0,0,255))

  cv2.imshow("dt", h)

def zoning(imb):

  cut = 7
  imHand = binaryCrop(imb)

  imWidth = imHand.shape[1]
  imHeight = imHand.shape[0]

  stepH = imHeight/cut -1
  stepW = imWidth/cut -1

  if(stepH < 1):
    stepH = 1

  if (stepW < 1):
    stepW = 1

  if imWidth <= cut or imHeight <= cut:
    return None

  density = []
  
  x = 1

  for i in range(0,imWidth-stepW,stepW):

    if x>cut:
      continue

    x = x+1
    y = 1

    for j in range(0,imHeight-stepH,stepH):

      if y>cut :
        continue

      y = y+1

      zone = imHand[j:j+stepH,i:i+stepW]
      zoneSize = zone.shape[0]*zone.shape[1]

      density.append(float(np.count_nonzero(zone))/float(zoneSize))

  
  if(len(density)!= cut*cut):
    print "Cut is too large for ROI"
    return None
    
  return density

def binaryCrop(imb):

  vertInd = np.where(np.argmax(imb,axis = 0)>0)
  y = vertInd[0][0] if vertInd[0].size else 0
  y2 = vertInd[0][-1] if vertInd[0].size else imb.shape[0]

  horInd = np.where(np.argmax(imb,axis = 1)>0)
  x = horInd[0][0] if horInd[0].size else 0
  x2 = horInd[0][-1] if horInd[0].size else imb.shape[1]
  
  crop = imb[x:x2,y:y2]
  return crop



if __name__ == '__main__' :
  cv2.namedWindow("Debug")
  cv2.namedWindow("Result")

  img_result, img_debug, density = process(loadSample(im))

  cv2.imshow("Debug", img_debug)
  cv2.imshow("Result", img_result)

  debugThresh(loadSample(im))

  cv2.waitKey(0)
