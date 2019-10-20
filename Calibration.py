import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.mgrid[0:7,0:6].T.reshape(-1,2)
print(objp)
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret:
      objpoints.append(objp)
      corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
      print(corners2)
      print(corners2.shape,objp.shape)
      imgpoints.append(corners2)
      # Draw and display the corners
      img = cv2.line(img,(0,0),(10,10),(0,0,255))
      img = cv2.line(img,(100,30),(10,10),(0,255,255))
      img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
      cv2.imshow('img',img)
      cv2.waitKey(20000)
