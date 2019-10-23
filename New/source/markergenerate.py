import cv2
from cv2 import aruco
from cv2.aruco import *
import os
import numpy as np
import glob
import math
from objloader_simple import *
import argparse
K = np.array([[517, 0, 309], [0, 379.5, 178], [0, 0, 1]],dtype=np.float32)
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])

distCoeffs = (0,0,0,0)
markerSize = 7.9  # centimeters

def extract_RT(RT):
    x1,x2,x3 = RT
    R = np.array([x1[:2],x2[:2],x3[:2]])
    T = np.array([x1[2],x2[2],x3[2]])
    return R,T

def get_relative_rt(H1, H2):
    H1_t = np.transpose(H1)
    print(H1_t)
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

def get_camera_pose(K,  corner):
    h,w = 200,200 #centi meter
    world_corner = np.float32([[0, 0], [w-1, 0], [w - 1, h - 1], [0, h-1]]).reshape(-1, 1, 2)
    # print(world_corner)
    homography,mask = cv2.findHomography(world_corner,corner)
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

if __name__=='__main__':
    md1, md2 = create_markers()
    
    # vd = cv2.VideoCapture('videos/video_1.mp4')
    vd = cv2.VideoCapture(0)
    
    # base_img = cv2.imread('markers/marker6_1.png')
    obj1 = OBJ('../models/fox.obj', swapyz=True)  
    obj2 = OBJ('../models/rat.obj', swapyz=True)
    play_using_aruco(md1,md2,vd)
    
