import os,glob,math,argparse
import cv2
import numpy as np
import Motion
from objloader_simple import *

### ======================================================================================================
### Get Intrinsic Matrix K
### ======================================================================================================

def getGrayImage(fname,shape):
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, shape)
	return gray

def getK():
	setDirec = input()
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((7*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	images = glob.glob('CalibrationImages/Set{0}/*.jpg'.format(setDirec))
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

def getArea(ls):
	a = ls[0]
	b = ls[1]
	c = ls[2]
	d1 = a-b
	d2 = c-b
	return np.linalg.norm(np.cross(d1,d2))

### ======================================================================================================
### Homography & Projection Matrix Functions
### ======================================================================================================

def getHomographyFromMatched(matches,kp1,kp2):
	if len(matches[:,0]) >= 4:
		src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
	return H,masked

def projection_matrix(camera_parameters, homography):
		"""
		From the camera calibration matrix and the estimated homography compute the 3D projection matrix
		"""
		# Compute rotation along the x and y axis as well as the translation
		homography = homography * (-1)
		rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
		col_1 = rot_and_transl[:, 0]
		col_2 = rot_and_transl[:, 1]
		col_3 = rot_and_transl[:, 2]
		# normalise vectors
		l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
		rot_1 = col_1 / l
		rot_2 = col_2 / l
		translation = col_3 / l
		# compute the orthonormal basis
		c = rot_1 + rot_2
		p = np.cross(rot_1, rot_2)
		d = np.cross(c, p)
		rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
		rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
		rot_3 = np.cross(rot_1, rot_2)
		# finally, compute the 3D projection matrix from the model to the current frame
		projection = np.stack((rot_1, rot_2, rot_3, translation)).T
		# print(projection)
		return np.dot(camera_parameters, projection)

def getMatches(desListi, desListj):
	matches = bf.knnMatch(desListi, desListj, k=2)
	good = []
	for m in matches:
		if m[0].distance < 0.5*m[1].distance:
			good.append(m)
	matches = np.asarray(good)
	return len(matches), matches

### ======================================================================================================
### Rendering the Object Functions
### ======================================================================================================

def hex_to_rgb(hex_color):
		hex_color = hex_color.lstrip('#')
		h_len = len(hex_color)
		return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
	""" Render a loaded obj model into the current video frame """
	try:
		vertices = obj.vertices
		scale_matrix = np.eye(3) * 3
		h, w = model.shape
		for face in obj.faces:
				face_vertices = face[0]
				points = np.array([vertices[vertex - 1] for vertex in face_vertices])
				points = np.dot(points, scale_matrix)
				# render model in the middle of the reference surface. To do so,
				# model points must be displaced
				points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
				dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
				imgpts = np.int32(dst)
				if color is False:
						cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
				else:
						color = hex_to_rgb(face[-1])
						color = color[::-1]  # reverse
						cv2.fillConvexPoly(img, imgpts, color)
	except:
		print("Unable to render the image")
	finally:
		return img
	
# def get_homography(frame, model, matches, kp_model, kp)	
def getProjectionAndRender(frame, model,matches,kp_model, kp_frame, projection, homography):
	# if len(matches) > MIN_MATCHES:
	# 	homography,_ = getHomographyFromMatched(matches,kp_model,kp_frame)
		## IF RECTANGLE
		# if args.rectangle: 
		h, w = model.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, homography)
		frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
		## IF HOMOGRAPHY EXISTS
		if homography is not None: 
			try:
					projection = projection_matrix(camera_parameters, homography)  
					projection = np.matmul(projection,position)
					frame = render(frame, obj, projection, model, False)
			except:
					print('cannot render object')
	# else:
	# 	print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))
	# 	frame = render(frame, obj, projection, model, False)
		return projection,frame

def get_smoothened_homo(H_old,H_new, alpha):
	return H_old*(1-alpha) + H_new*(alpha)

def percentageChange(q1,q2):
	x = 100*(q1-q2)/q2
	if x>0:
		return x
	else :
		return -x

### ======================================================================================================
### Intilizae Model, Marker and Descriptors
### ======================================================================================================

## ======================== Arguments ========================== ##

parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
args = parser.parse_args()

## ======================== Camera Properties ========================== ##

homography1 = None 
projection1 = None

homography2 = None
projection2 = None

# camera_parameters = getK()
camera_parameters = np.array([[517.23, 0, 309.16], [0, 379.5, 177.93], [0, 0, 1]],dtype=np.float32)

MIN_MATCHES = 8

## ==================== Marker and Model Load ======================== ##

## Load Marker
model1 = cv2.imread('../markers/marker1/VisualMarker.png',0)
model2 = cv2.imread('../markers/marker1/VisualMarker1.png',0)

# Load 3D model from OBJ file
obj = OBJ('../models/fox.obj', swapyz=True)  

## ======================== ORB ========================== ##

# # create ORB keypoint detector
# orb = cv2.ORB_create()
# # create BFMatcher object based on hamming distance  
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # Compute model keypoints and its descriptors
# kp_model, des_model = orb.detectAndCompute(model, None)

## ======================= SIFT ========================== ##

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
kp_model1, des_model1 = sift.detectAndCompute(model1,None)
kp_model2, des_model2 = sift.detectAndCompute(model2,None)

### ======================================================================================================
### Streaming On Each Frame -- Part 4
### ======================================================================================================

## ================ Video Capture Initialize ============= ##

cap = cv2.VideoCapture('../Test4_{0}.mp4'.format(input()))
# cap = cv2.VideoCapture(0)

## ===================== Start Streaming ================= ##

position = np.identity(4)
initialPoint = np.array([0,0,0])
finalPoint = np.array([0,-500,0])
step = Motion.getMotionStep(initialPoint,finalPoint,30)

homo1 = None
homo2 = None


h, w = model1.shape
pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

h, w = model2.shape
pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

area1 = math.inf
area2 = math.inf

while True:
	ret, frame = cap.read()
	if not ret:
		print("Unable to capture video or End of Video")
		break 
	
	## == Choose Frame for Detection == ##
	detectframe = frame ## Normal Frame For Detection	
	detectframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) ## Gray Frame For Detection
	_,detectframe = cv2.threshold(detectframe, 200, 255, cv2.THRESH_BINARY) ## Black and White Frame Detection

	## == find and draw the keypoints of the frame ==##
	# kp_frame, des_frame = orb.detectAndCompute(frame, None)  ## ORB
	kp_frame, des_frame = sift.detectAndCompute(detectframe, None)    ## SIFT
	
	# match frame descriptors with model descriptors
	# matches = bf.match(des_model, des_frame)           ## ORB
	_, matches1 = getMatches(des_model1, des_frame)    ## SIFT
	_, matches2 = getMatches(des_model2, des_frame)    ## SIFT

	# sort them in the order of their distance
	# the lower the distance, the better the match
	# matches = sorted(matches, key=lambda x: x.distance)
	# print(matches)

	## == Update Position == ##
	position = np.matmul(position,step)

	alpha = 0.25
	count_1=0
	count_2=0


	if len(matches2)>MIN_MATCHES:
		homography,_ = getHomographyFromMatched(matches2,kp_model2,kp_frame)
		if homography is not None:
			dst = cv2.perspectiveTransform(pts2, homography)
			AreaQuad = getArea(dst[0:3])+getArea(dst[1:])
			if homography2 is not None and percentageChange(AreaQuad,area2)<20:
				homography2 = get_smoothened_homo(homography,homography2,alpha)
				print(AreaQuad,area2)
				area2 = AreaQuad
			else:
				homography2 = homography
			count_1 = 0
	else:
		count_1 += 1
		if (count_1==20):
			count_1 = 0
			homography2 = None
	if homography2 is not None:
		projection2,frame = getProjectionAndRender(frame, model2, matches2, kp_model2, kp_frame, projection2, homography2)

	if len(matches1)>MIN_MATCHES:
		homography,_ = getHomographyFromMatched(matches1,kp_model1,kp_frame)
		if homography is not None:
			dst = cv2.perspectiveTransform(pts1, homography)
			AreaQuad = getArea(dst[0:3])+getArea(dst[1:])	
			if homography1 is not None and percentageChange(AreaQuad,area1)<33:
				homography1 = get_smoothened_homo(homography,homography1,alpha)
				area1 = AreaQuad
			else:
				homography1 = homography
			count_2 = 0
	else:
		count_1 += 1
		if (count_1==20):
			count_1 = 0
			homography1 = None
	if homography1 is not None:
		projection1,frame = getProjectionAndRender(frame, model1, matches1, kp_model1, kp_frame, projection1, homography1)

	## == Show Frame == ##
	cv2.imshow('frame', frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
		
cap.release()
cv2.destroyAllWindows()
