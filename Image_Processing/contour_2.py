#Moments

import cv2
import numpy as np

img = cv2.imread('approx.jpg',0)
ret,thresh = cv2.threshold(img,127,255,0)
_,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
# calcul area of a contour

area = cv2.contourArea(cnt)

# Calcul the perimeter of a contour
perimeter = cv2.arcLength(cnt,True)

#Contour Approximation

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
approx_x=approx[:len(approx)-1]
approx_y=approx
print approx
for index, point in enumerate(approx):
	print point
	cv2.line(img, (approx[index-1][0][0],approx[index-1][0][1]), (approx[index][0][0],approx[index][0][1]), (255,0,0),40)
	
		#print "hello"
		#print approx[index-1][0][0],approx[index-1][0][1], approx[index][0][0],approx[index][0][1]

#cv2.line(img, (5,2), (50, 125), (0,0,255),20)
cv2.imshow("approx", img)
cv2.waitKey(0)


# convex hull
hull = cv2.convexHull(cnt)

cv2.drawContours(img,hull,-1,(0,255,0),3)

cv2.imshow("hull", img)
cv2.waitKey(0)
#bouding rect
x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow("rect", img)
cv2.waitKey(0)
# rotated rect
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(0,0,255),2)
cv2.imshow("rotated", img)
cv2.waitKey(0)



print "area:",area
print "perimeter", perimeter
#print "moments", M