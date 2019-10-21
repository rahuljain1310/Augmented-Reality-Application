import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.

def getPointsFromFrame(fr,objpoints,imgpoints):
	gray = cv2.cvtColor(fr ,cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(gray, (7,7),None)
	if ret == True:
		objpoints.append(objp)
		corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		imgpoints.append(corners2)
		img = cv2.drawChessboardCorners(fr, (7,7), corners2, ret )
		return img
	return fr


vd = cv2.VideoCapture(0)
_,fr = vd.read()
h,w,d = fr.shape
j = 0
while True:
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	for i in range(10):
		ret, fr = vd.read()
		if ret:
			gr = getPointsFromFrame(fr,objpoints,imgpoints)
			cv2.imshow('Image',gr)
			if cv2.waitKey(100) == 27:
				cv2.imwrite("Image_{0}.jpg".format(j),fr)
				j += 1
	try:
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w,h), None,None)
		if ret:
			print("Camera Matrix")
			print(mtx)
		else:
			print("No Solution")
	except:
		print("No Solution Found")