import os
import argparse
import cv2
import numpy as np
import glob
from objloader_simple import *

dir_markers = os.path.join(os.pardir,'markers')
dir_chess = os.path.join(os.pardir,'chessboards')
dir_objects = os.path.join(os.pardir,'objects')

def capture_boards():
    vd = cv2.VideoCapture(0)
    img_counter = 1
    while(True):
        ret,frame = vd.read()
        cv2.imshow('Capture chess',frame)
        k = cv2.waitKey(10)
        if (k is not -1):
            print(k)
        # print(k)
        if k%256 == 27:
        # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256==32:
        # SPACE pressed
            print("Space hitting")
            img_name = os.path.join(dir_chess,'{}.png'.format(img_counter))
            print(img_name)
            cv2.imwrite(img_name, frame)
            img_counter += 1
    vd.release()


    # capture_boards()

def capture_chessboard():
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model = cv2.imread(os.path.join(dir_name, 'reference/model.jpg'), 0)
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture(0)

def getGrayImage(fname,shape):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, shape)
	return gray

def getK():
# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((7*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('CalibrationImages/*.jpg')
	Shape = None

	for fname in images:
		gray = getGrayImage(fname,(1008,756))
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
		print("Camera Matrix")
		print(mtx)
	else:
		print("No Solution Found")

	cv2.destroyAllWindows()

if __name__=='__main__':
    capture_boards()

