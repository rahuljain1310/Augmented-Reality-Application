import os,glob,math,argparse
import cv2
import numpy as np

def getGrayImage(fname,shape):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, shape)
	return gray

def getK():
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((7*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('CalibrationImages/Set4/*.jpg')
	Shape = None

	for fname in images:
		gray = getGrayImage(fname,(640,352))
		Shape = gray.shape[::-1]
		ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
		if ret == True:
			print(fname)
			objpoints.append(objp)
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)
			# Draw and display the corners
			img = cv2.drawChessboardCorners(gray, (7,7), corners2, ret )
			cv2.imshow('img',gray)
			cv2.waitKey(2000)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, Shape, None,None)
	if ret:
		print("Image Size:")
		print(Shape)
		print("Camera Matrix:")
		print(mtx)
		return mtx
	else:
		print("No Solution Found")
		return None
	cv2.destroyAllWindows()
