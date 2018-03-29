#Contour : Python list of all the contours in the image
# Each individual contour is a Numpy array of (x,y) 
#coordinates of boundary points of the object
import numpy as np
import cv2

im = cv2.imread('cni.jpeg')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
_,contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#cv2.CHAIN_APPROX_NONE, all the boundary points are stored
# cv2.CHAIN_APPROX_SIMPLE removes all redundant points and compresses the contour

#Parameter: img, list_contours, -1 : all, color, thickness
#cv2.drawContours(im, contours, -1, (0,255,0), 3)

#To draw an individual contour, say 4th contou
#cv2.drawContours(im, contours, 3, (0,255,0), 3)

# Or 
cnt = contours[4]
cv2.drawContours(im, [cnt], 0, (0,255,0), 3)

cv2.imshow("contour",im)

cv2.waitKey(0)