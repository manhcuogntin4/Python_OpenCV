#Canny Edge Detection
# 1 Noise reduction with Gaussain
# 2 Finding Intensity Gradient of image with Sobel in horizontal and vertical
# 3 Non-maximum Suppression
# Thresholding with MaxVal and MinVal
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg',0)
edges = cv2.Canny(img,50,400)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()