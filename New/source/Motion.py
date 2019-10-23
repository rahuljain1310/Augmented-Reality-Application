import numpy as np
import math

def getCentroid(lp):
  assert lp.shape[0] == 4
  x = np.array(lp)
  return np.mean(x,axis=0)

def getPlane(ls):
  p1 = ls[0]
  p2 = ls[1]
  p3 = ls[2]
  d1 = p2-p1
  d2 = p3-p1
  a,b,c = np.cross(d1,d2)
  d = (- a * p1[0] - b * p1[1] - c * p1[2])
  planeEquation = np.array([a,b,c,d],dtype=float)
  return planeEquation/math.sqrt(a*a+b*b+c*c)

def getYcoordinate(plane, x, z=0):
  a,b,c,d = plane
  return (d-a*x-c*z)/b

def getMotionStep(intialPoint, finalPoint, step):
  dstVec = finalPoint-intialPoint
  dst = np.linalg.norm(dstVec)
  stepTranslationVec = dstVec*(step/dst)
  RT = np.identity(4)
  RT[:3,3] = stepTranslationVec
  return RT

def getFinalPoint(ls,x,z=0):
  plane = getPlane(ls)
  y = getYcoordinate(plane,x,z)
  return np.array([x,y,z,1])

def getReflectionFromPlane(plane,incident):
  n = plane[0:3]
  return incident-2*n.dot(incident)*n

# ls = np.array([[-1,2,1], [0,-3,2], [1,1,-4]])
# plane = getPlane(ls)
# incident = np.array([1,0,0])
# print(getReflectionFromPlane(plane,incident))

  

