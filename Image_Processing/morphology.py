# -*- coding: utf-8 -*-
import cv2
import numpy as np

#erosion
# The kernel slides through the image (as in 2D convolution). 
# A pixel in the original image (either 1 or 0) will be considered 1 only 
# if all the pixels under the kernel is 1, otherwise it is eroded (made to zero).
# all the pixels near boundary will be discarded depending upon the size of kernel

import cv2
import numpy as np

img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)

cv2.imshow("erosion", erosion)
cv2.waitKey(0)

# Dilation

# It is just opposite of erosion. 
# Here, a pixel element is ‘1’ if atleast one pixel under the kernel is ‘1’

dilation = cv2.dilate(img,kernel,iterations = 1)

cv2.imshow("dilation", dilation)
cv2.waitKey(0)

# Opening
#Opening is just another name of erosion followed by dilation.
#It is useful in removing noise

img = cv2.imread('j_open.png',0)
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow("open", opening)
cv2.waitKey(0)

#Closing
#Closing is reverse of Opening,
# Dilation followed by Erosion. 
#It is useful in closing small holes inside the foreground objects,
# or small black points on the object
img = cv2.imread('j_closing.png',0)
kernel = np.ones((5,5),np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow("closing", closing)
cv2.waitKey(0)

#Morphological Gradient
#It is the difference between dilation and erosion of an image.
#The result will look like the outline of the object.
img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("gradient", gradient)
cv2.waitKey(0)

#Top hat
#It is the difference between input image and Opening of the image. 
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
cv2.imshow("top hat", tophat)
cv2.waitKey(0)

#Black hat
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow("black hat", blackhat)
cv2.waitKey(0)