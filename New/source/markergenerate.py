import cv2
from cv2 import aruco
from cv2.aruco import *
import get_intrinsic
import os
import numpy as np
import glob
import math
from objloader_simple import *
import shapely
from shapely.geometry import Polygon,LineString,Point

# from Assignment4 import * 
import argparse
# K = np.array([[517, 0, 309], [0, 379.5, 178], [0, 0, 1]],dtype=np.float32)
K = getK()
# K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

distCoeffs = (0,0,0,0)
markerSize = 7.9  # centimeters
position = np.identity(4)
sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
obj1 = OBJ('../models/fox.obj', swapyz=True)  
pixelCmRatio = 9.5/(500-16)

def init_():
		model2 = cv2.imread('../markers/marker1/VisualMarker1.png',0)
		model1 = cv2.imread('../markers/marker1/VisualMarker.png',0)
		# model1 = cv2.rotate(model1, cv2.rotate)
		# model1.reshape(400,500)
		# model2.reshape(400,500)
		# print(model1.shape)
		# print(model2.shape)
		sift = cv2.xfeatures2d.SIFT_create()
		bf = cv2.BFMatcher()
		kp_model1, des_model1 = sift.detectAndCompute(model1,None)
		kp_model2, des_model2 = sift.detectAndCompute(model2,None)
		return kp_model1, kp_model2, des_model1, des_model2, model1.shape, model2.shape
		# Load 3D model from OBJ file

		## ======================== ORB ========================== ##

		# # create ORB keypoint detector
		# orb = cv2.ORB_create()
		# # create BFMatcher object based on hamming distance  
		# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
		# # Compute model keypoints and its descriptors
		# kp_model, des_model = orb.detectAndCompute(model, None)

		## ======================= SIFT ========================== ##
		


def get_keypoints(cap,des_model1,des_model2):
		ret, frame = cap.read()
		if (not ret):
				print("Unable to capture video or End of Video")
				return None
		detectframe = frame
		## Normal Frame For Detection	
		detectframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) ## Gray Frame For Detection
		_,detectframe = cv2.threshold(detectframe, 200, 255, cv2.THRESH_BINARY) ## Black and White Frame Detection

	## == find and draw the keypoints of the frame ==##
	# kp_frame, des_frame = orb.detectAndCompute(frame, None)  ## ORB
		kp_frame, des_frame = sift.detectAndCompute(detectframe, None)    ## SIFT
		
		# match frame descriptors with model descriptors
		# matches = bf.match(des_model, des_frame)           ## ORB
		_, matches1 = getMatches(des_model1, des_frame)    ## SIFT
		_, matches2 = getMatches(des_model2, des_frame) 
		return matches1, matches2, kp_frame, des_frame,frame

def getMatches(desListi, desListj):
	matches = bf.knnMatch(desListi, desListj, k=2)
	good = []
	for m in matches:
		if m[0].distance < 0.5*m[1].distance:
			good.append(m)
	matches = np.asarray(good)
	return len(matches), matches

def getHomographyFromMatched(matches,kp1,kp2):
	if len(matches[:,0]) >= 4:
		src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
		H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
	return H,masked


def extract_RT(RT):
		x1,x2,x3 = RT
		R = np.array([x1[:3],x2[:3],x3[:3]])
		T = np.array([x1[3],x2[3],x3[3]])
		return R,T

def get_relative_rt(H1, H2):
		H1_t = np.transpose(H1)
		# print(H1_t)
		RT = np.matmul(np.matmul(np.linalg.inv(np.matmul(H1_t, H1)),H1_t),H2)
		return RT

def project_onto_image():
		pass

def create_markers():
		j=2
		markers_dict_1 = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
		markers_dict1= custom_dictionary_from(j,6,markers_dict_1)
		markers_dict_2 = aruco.getPredefinedDictionary(aruco.DICT_7X7_50)
		markers_dict2 = custom_dictionary_from(j,7,markers_dict_2)

		markers1 = list(aruco.drawMarker(markers_dict1,i+1,200) for i in range(1))
		markers2 = list(aruco.drawMarker(markers_dict2,i+1,200) for i in range(1))
		
		for i in range(len(markers1)):
				cv2.imwrite(os.path.join('markers','marker6_')+str(i+1) + '.png',markers1[i])
		for i in range(len(markers2)):
				cv2.imwrite(os.path.join('markers','marker7_')+str(i+1)+'.png', markers2[i])
		return markers_dict1, markers_dict2
		# return markers
# def play_using_aruco(md1,md2,vd):

def play_using_aruco(md1,md2,vd):
		model_shape = (20,20)
		c1 = None
		c2 = None
		nfc1 = 0
		nfc2 = 0
		nfc_limit = 20
		pm1 = None
		pm2 = None
		while(True):
				ret, frame = vd.read()
				# print(frame.shape)
				# print ret
				if  ret:
						corners1, ids, rejectedpts = detectMarkers(frame,md1)
						if (len(corners1)>=1):
								c1 = corners1[0]
								nfc1 = 0
						else:
								nfc1 +=1
								if (nfc1==nfc_limit):
										c1=None
										pm1 = None
										nfc1 = 0

						# corners2,_,_ = detectMarkers(frame,md2)
						corners2, ids2,rp2 = detectMarkers(frame,md2)
						if (len(corners2)>=1):
								c2 = corners2[0]
								nfc2=0
						else:
								nfc2 +=1
								if (nfc2==nfc_limit):
										c2=None
										nfc2 = 0

						if (c1 is not None and c2 is not None):
								# print(5)
								pm2 = get_camera_pose(K,c2)
								if pm1 is None:
										pm1 = get_camera_pose(K,c1)
								else:
										pm1 = get_pm(pm1,pm2)
								
								
								# RT_rel = get_relative_rt(pm1,pm2)
								frame = drawDetectedMarkers(frame,[c1,c2])
								# frame = drawDetectedMarkers(frame,c2)
								frame = render(frame,obj1,pm1,model_shape)
								# frame = render(frame, obj2,pm2,model_shape)
						
						cv2.imshow('corner',frame)
						# print(corners,ids)
						cv2.waitKey(1000//30)

def play_using_aruco(kpm1,des1,vd,model_shape):
		# model_shape = (20,20)
		c1 = None
		c2 = None
		nfc1 = 0
		nfc2 = 0
		nfc_limit = 20
		pm1 = None
		pm2 = None
		while(True):
				ret, frame = vd.read()
				# print(frame.shape)
				# print ret
				if  ret:
						corners1, ids, rejectedpts = detectMarkers(frame,md1)
						if (len(corners1)>=1):
								c1 = corners1[0]
								nfc1 = 0
						else:
								nfc1 +=1
								if (nfc1==nfc_limit):
										c1=None
										pm1 = None
										nfc1 = 0

						# corners2,_,_ = detectMarkers(frame,md2)
						corners2, ids2,rp2 = detectMarkers(frame,md2)
						if (len(corners2)>=1):
								c2 = corners2[0]
								nfc2=0
						else:
								nfc2 +=1
								if (nfc2==nfc_limit):
										c2=None
										nfc2 = 0

						if (c1 is not None and c2 is not None):
								# print(5)
								pm2 = get_camera_pose(K,c2)
								if pm1 is None:
										pm1 = get_camera_pose(K,c1)
								else:
										pm1 = get_pm(pm1,pm2)
								
								
								# RT_rel = get_relative_rt(pm1,pm2)
								frame = drawDetectedMarkers(frame,[c1,c2])
								# frame = drawDetectedMarkers(frame,c2)
								frame = render(frame,obj1,pm1,model_shape)
								# frame = render(frame, obj2,pm2,model_shape)
						
						cv2.imshow('corner',frame)
						# print(corners,ids)
						cv2.waitKey(1000//30)


def get_dist(a):
		return math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
def get_pm(pm, pm2,timestep=1000/30, velocity = 0.1):
		R1,T1 = extract_RT(pm)
		R2,T2 = extract_RT(pm2)
		time_required = get_dist((T2 - T1))/velocity
		return ((time_required - timestep)*pm + timestep*pm2)/time_required

def Hsmoothening(H_old, H_new, alpha):
		return  alpha*H_old + (1-alpha)*H_new


def detect_markers(md1, md2, vd):
		model_shape  = (20,20)
		while(True):
				ret, frame = vd.read()
				# print(frame.shape)
				# print ret
				if  ret:
						corners, ids, rejectedpts = detectMarkers(frame,md1)
						# corners2,_,_ = detectMarkers(frame,md2)
						corners2 = []
						# corners. 
						# print(corners)
						# print(corners)
						# cv2.imshow('frame',frame)
						lengt = len(corners) + len(corners2)
						if lengt==1:
								if len(corners)==0:
										corners = corners2
								projection_matrix1 = get_camera_pose(K,corners[0])
								frame = drawDetectedMarkers(frame,corners)
								frame = render(frame, obj2,projection_matrix1,model_shape)
								# cv2.imshow('corner',frame)
						elif lengt>1:
								# print(corners[0])

								# rvecs, tvecs, _objPoints = estimatePoseSingleMarkers(corners, markerSize, K, distCoeffs)
								# R = cv2.Rodrigues(rvecs[0])[0]
								# T = np.transpose(tvecs[0])
								# # print(R)
								# projection_matrix = np.concatenate((np.matmul(K,R),np.matmul(K,T)),axis=1 )
								# # print(projection_matrix)
								# print('hello')
								projection_matrix1 = get_camera_pose(K,corners[1])
								projection_matrix2 = get_camera_pose(K,corners[0])
								RT_rel = get_relative_rt(projection_matrix1,projection_matrix2)
								# print(projection_matrix)
								# if  rvecs, tvecs 
								# print(rvecs,tvecs)
								frame = drawDetectedMarkers(frame,corners)
								frame = render(frame,obj1,projection_matrix1,model_shape)
								frame = render(frame, obj2,projection_matrix2,model_shape)
								# for rvec, tvec in zip(rvecs,tvecs):
										# frame = drawAxis(frame, K, distCoeffs, rvec, tvec, 5)     
								
						cv2.imshow('corner',frame)
						# print(corners,ids)
						cv2.waitKey(1000//30)

def get_camera_pose(K,  corner, model_shape):
		h,w = model_shape #centi meter
		world_corner = np.float32([[0, 0], [w-1, 0], [w - 1, h - 1], [0, h-1]]).reshape(-1, 1, 2)
		# print(world_corner)
		homography,mask = cv2.findHomography(world_corner*pixelCmRatio,corner)
		projection = projection_matrix(K, homography)  
		return projection

def estimate_camera_pose(K, base_img_coords, img_coords, ratio):
		pass

def projection_matrix(camera_parameters, homography):
		"""
		From the camera calibration matrix and the estimated homography
		compute the 3D projection matrix
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

def render(img, obj, projection, model, color=False):
		"""
		Render a loaded obj model into the current video frame
		"""
		vertices = obj.vertices
		scale_matrix = np.eye(3) * 3
		h, w = model

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

		return img
def get_t_homo(h1):
	x1,x2,x3 = h1
	return np.array([x1[2],x2[2],x3[2]])

def get_dist(t1,t2):
	t = t1-t2
	return math.sqrt(t[0]*t[0] + t[1]*t[1] + t[2]*t[2])

def get_mid_homo(h1,h2, velocity = 0.05,timespan = 1000//30):
	total_dist = get_dist(get_t_homo(h1),get_t_homo(h2))
	tt = total_dist/velocity
	if (tt<=timespan):
		return h2
	return (h1*(tt-timespan) + h2*(timespan))/tt

def play_using_sift(kpm1,kpm2,des1,des2,vd,model_shape):
		# fourcc = cv2.VideoWriter_fourcc(*'XVID')
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 	
		vd_write = cv2.VideoWriter('video_out_test4_3.mp4',fourcc,15,(640,352))
		homo1 = None
		homo2 = None
		MIN_MATCHES = 8
		alpha_h= 0.05
		h,w= model_shape[0]
		# print(h,w)
		h2,w2 = model_shape[1]
		homo_cur = None
		
		pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		pts2 = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)

		while(True):
				X = get_keypoints(vd,des1,des2)
				# print(1)
				if X is None:
						break
				else:
						matches1,matches2, kp_frame, des_frame,frame = X
				if (len(matches1)>MIN_MATCHES):
						homo1_t,marked = getHomographyFromMatched(matches1, kpm1,kp_frame)
						if homo1 is None:
								homo1 = homo1_t
								homo_cur = homo1
								homo_1 = homo1
						else:
								homo1 = np.array((1-alpha_h)*homo1 + (alpha_h)*homo1_t)
								# 
								dst = cv2.perspectiveTransform(pts1, homo1)
								frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
								# 
								# print(homo2)
								# pm1 = projection_matrix(K, homo1)  
								# print(pm2)
								# frame = render(frame,obj1,pm1,model_shape)

				# print(len(matches2))
				if (len(matches2) > MIN_MATCHES):
						homo2_t,_masked = getHomographyFromMatched(matches2, kpm2,kp_frame)
						# print(homo2_t)
						if homo2 is None:
								homo2 = homo2_t
						else:
								homo2 = np.array((1-alpha_h)*homo2 + (alpha_h)*homo2_t)
								# 
								dst = cv2.perspectiveTransform(pts2, homo2)
								frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
								# 
								# print(homo2)
								# pm2 = projection_matrix(K, homo2)  
								# print(pm2)
								# frame = render(frame,obj1,pm2,model_shape)
				if homo_cur is not None:
						if homo2 is not None:
							homo_cur = get_mid_homo(homo_cur,homo2)
						pm = projection_matrix(K,homo_cur)
						
						frame = render(frame,obj1,pm,model_shape[0])
						

			
				
				cv2.imshow('frame',frame)
				frame = np.asarray(frame,dtype=np.uint8)
				vd_write.write(frame)
				cv2.waitKey(1000//30)

def stay_using_sift(kpm1,des1,vd,model_shape,alpha_h):
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 	
		vd_write = cv2.VideoWriter('video_out_testvideo.mp4',fourcc,15,(640,352))
		homo1 = None
		# homo2 = None
		MIN_MATCHES = 8
		# alpha_h= 0.05
		h,w= np.array(model_shape)
		homo_cur = None
		
		pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		# pts2 = np.float32([[0, 0], [0, h2 - 1], [w2 - 1, h2 - 1], [w2 - 1, 0]]).reshape(-1, 1, 2)
		homo_count = 0
		homo_max_count = 20
		while(True):
				ret, frame = vd.read()
				if (not ret):
						print("Unable to capture video or End of Video")
						break
				detectframe = frame
				## Normal Frame For Detection	
				detectframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) ## Gray Frame For Detection
				_,detectframe = cv2.threshold(detectframe, 200, 255, cv2.THRESH_BINARY) ## Black and White Frame Detection

		
				kp_frame, des_frame = sift.detectAndCompute(detectframe, None)    ## SIFT
		
				# matches = bf.match(des_model, des_frame)           ## ORB
				_, matches1 = getMatches(des1, des_frame)    ## SIFT
	
				if (len(matches1)>MIN_MATCHES):
						homo1_t,marked = getHomographyFromMatched(matches1, kpm1,kp_frame)
				else:
						homo1_t = None

				if homo1 is None and homo1_t is not None:
						homo1 = homo1_t
						
					
				if homo1 is not None and homo1_t is not None:
						homo1 = np.array((1-alpha_h)*homo1 + (alpha_h)*homo1_t)
						homo_count = 0
				else:
						homo_count += 1
						if (homo_count > homo_max_count):
								homo1 = None
								homo_count = 0

				if homo1 is not None:
								# 
						dst = cv2.perspectiveTransform(pts1, homo1)
						frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
						# 
						# print(homo2)
						pm= projection_matrix(K, homo1)
						RT = np.matmul(np.linalg.inv(K),pm)
						R,T =extract_RT(RT)
						t1,t2,t3 = (pixelCmRatio*T)
						T_n = np.array([[t1],[t2],[t3]])
						# # print(R)
						# print(T)
						RT = np.concatenate((R,T_n),axis=1)
						print(RT)
						frame = render(frame,obj1,pm,model_shape)

				cv2.imshow('frame',frame)
				frame = np.asarray(frame,dtype=np.uint8)
				vd_write.write(frame)
				cv2.waitKey(1000//30)


if __name__=='__main__':
		md1, md2 = create_markers()
		kpm1, kpm2, des1, des2,m1s,m2s = init_()
		
		# vd = cv2.VideoCapture('videos/video_1.mp4')
		# vd = cv2.VideoCapture(0)
		vd = cv2.VideoCapture('../Test4_3.mp4')
		vd1 = cv2.VideoCapture('../video1.mp4')
		# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #

		# vd_write = cv2.VideoWriter('video_out_test4_3.mp4',fourcc,30,(640,352))
		# while(True):
		# 	ret,frame = vd.read()
		# 	if ret:
		# 		vd_write.write(frame)
		# play_using_sift(kpm1,kpm2,des1,des2,vd,(m1s,m2s))

		# base_img = cv2.imread('markers/marker6_1.png')
		obj1 = OBJ('../models/fox.obj', swapyz=True)  
		obj2 = OBJ('../models/rat.obj', swapyz=True)

		stay_using_sift(kpm1,des1,vd1,m1s,0.1)
		# play_using_aruco(md1,md2,vd)
		
