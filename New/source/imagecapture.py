import os
import argparse
import cv2
import numpy as np
import glob
import math
from objloader_simple import *

dir_markers = os.path.join(os.pardir,'markers')
dir_chess = os.path.join(os.pardir,'chessboards')
dir_objects = os.path.join(os.pardir,'objects')

MIN_MATCHES = 30

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

def main():
    """
    This functions loads the target surface image,
    """
    homography = None 

    camera_parameters = np.array([[517, 0, 309], [0, 379.5, 178], [0, 0, 1]],dtype=np.float32)
    # camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # camera_parameters = np.array([[1, 0, 2], [0, 2, 1], [0, 0, 1]],dtype=np.float32)

	# camera_parameters = np.array([])
    # create ORB keypoint detector
    orb = cv2.ORB_create()
    # create BFMatcher object based on hamming distance  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # load the reference surface that will be searched in the video stream
    # dir_name = os.getcwd()
    model = cv2.imread('../markers/marker1/VisualMarker.png', 0)
    # cv2.resize(model,(640,352))
    # Compute model keypoints and its descriptors
    kp_model, des_model = orb.detectAndCompute(model, None)
    # Load 3D model from OBJ file
    obj = OBJ('../models/fox.obj', swapyz=True)  
    # init video capture
    cap = cv2.VideoCapture('../video1.mp4')
    # cap = cv2.VideoCapture(0)
    

    while True:
        # read the current frame
        ret, frame = cap.read()
        print(frame.shape)
        if not ret:
            print("Unable to capture video")
            return 
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        matches = bf.match(des_model, des_frame)
        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)
        # print(matches)
        # compute Homography if enough matches are found
        if len(matches) > MIN_MATCHES:
            # differenciate between source points and destination points
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            # compute Homography
            homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if args.rectangle:
                # Draw a rectangle that marks the found model in the frame
                h, w = model.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                # project corners into frame
                dst = cv2.perspectiveTransform(pts, homography)
                # connect them with lines  
                frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
            # if a valid homography matrix was found render cube on model plane
            if homography is not None:
                # try:
                    # obtain 3D projection matrix from homography matrix and camera parameters
                    projection = projection_matrix(camera_parameters, homography)  
                    # project cube or model
                    frame = render(frame, obj, projection, model, False)
                    #frame = render(frame, model, projection)
                # except:
                #     print('cannot render object')
            # draw first 10 matches.
            if args.matches:
                frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:10], 0, flags=2)
            # show result
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()
    return 0

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


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

parser = argparse.ArgumentParser(description='Augmented reality application')
parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
args = parser.parse_args()

if __name__=='__main__':
    main()

