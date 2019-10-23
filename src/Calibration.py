import numpy as np
import cv2
import glob
from objloader import *
import os

dir_name = os.getcwd()
dir_models = os.path.join(dir_name,'reference')
dir_objs = os.path.join(dir_name,'models')

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

def render(img, obj, projection, model, color=False):
    """
    Render a loaded obj model into the current video frame
    """
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

    return img

def main():
    model_h = 