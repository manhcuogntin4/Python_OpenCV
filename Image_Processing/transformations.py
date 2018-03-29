import cv2
import numpy as np
from matplotlib import pyplot as plt

#scaling

img = cv2.imread('messi5.jpg')

res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

#OR

height, width = img.shape[:2]
res = cv2.resize(img,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

cv2.imshow("scalling", res)
cv2.waitKey(0)

cv2.destroyAllWindows()


# shifting 100->x, 50->y

rows,cols,_ = img.shape

M = np.float32([[1,0,100],[0,1,50]])
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Rotation

img = cv2.imread('messi5.jpg',0)
rows,cols = img.shape

M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
dst = cv2.warpAffine(img,M,(cols,rows))

cv2.imshow('img',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Affine

img = cv2.imread('drawing.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]]) # Need 3 points in original image
pts2 = np.float32([[10,100],[200,50],[100,250]]) # Need 3 points in output image

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

# Perpectif transforme

img = cv2.imread('sudokusmall.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[37,35],[186,28],[20,188],[194,193]]) # Need 4 points in the input image
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]]) # Need 4 points in the output image

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
