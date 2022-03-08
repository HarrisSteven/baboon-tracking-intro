import numpy as np
import cv2 as cv
cap = cv.VideoCapture('input.mp4')
cv.namedWindow("output", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv.namedWindow("bgsub", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions
cv.namedWindow("cvsol", cv.WINDOW_NORMAL)    # Create window with freedom of dimensions

prevFrame = np.zeros((540,960))
threshold = 100
count = 0
# totalFrame = np.zeros((540,960,3))
# totalSubFrame = np.zeros((540, 960, 3))
totalFrame = np.zeros((540,960))
totalSubFrame = np.zeros((540, 960))
isSub = True
skip = 20
orb = cv.ORB_create()
matcher = cv.BFMatcher()
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()

while True:
  ret, frame = cap.read()
  if frame is None:
    break

  frame = cv.resize(frame, (960, 540))                # Resize image
  frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  fgmask = fgbg.apply(frame)
  cv.imshow("cvsol", fgmask) 

  if count == 0:
    count += 1
    prevFrame = frame 
    continue

  # curFrameKeypoints, curFrameDescriptors = orb.detectAndCompute(frame,None)
  # prevFrameKeypoints, prevFrameDescriptors = orb.detectAndCompute(prevFrame,None)
  # matches = matcher.match(curFrameDescriptors,prevFrameDescriptors)

  # print("START")
  # for match in matches:
  #   queryPt = curFrameKeypoints[match.queryIdx].pt
  #   trainPt = prevFrameKeypoints[match.trainIdx].pt
  #   translation = (queryPt[0] - trainPt[0], queryPt[1] - trainPt[1])
  #   if(translation[0] < 5 and translation[0] > -5 and translation[1] < 5 and translation[1] > -5 and (translation[0] != 0 and translation[1] != 0)):
  #     print(translation)
  # print("END")

  cv.imshow("output", frame) 
  
  if(count % (skip*2) == 0):
    avgFrame = totalFrame / skip
    avgSub = totalSubFrame / skip
    
    subFrame = avgFrame - avgSub 
    totalFrame = np.zeros((540,960))
    totalSubFrame = np.zeros((540,960))

    cv.imshow("bgsub", subFrame) 

  # print(subFrame)
  #print(np.linalg.norm(subFrame))
  #print(np.amax(subFrame))

  prevFrame = frame

  count += 1
  if count % skip == 0:
    isSub = not isSub

  if isSub:
    totalSubFrame += frame

  else:
    totalFrame += frame

  # cv.imshow('Frame', frame)
  keyboard = cv.waitKey(30)
  if keyboard == 'q' or keyboard == 27:
    break

cap.release()
cv.destroyAllWindows()
