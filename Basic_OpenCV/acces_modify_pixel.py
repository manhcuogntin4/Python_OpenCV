import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg')
px = img[100,100]
# Display 3 channels of pixel
print px

# accessing only blue pixel
blue = img[100,100,0]
print blue

#You can modify the pixel values the same way

img[100,100] = [255,255,255]
print img[100,100]


img.item(10,10,2)
img.itemset((10,10,2),100)
img.item(10,10,2)



print img.shape
# Image size h*w
print img.size

# data type
print img.dtype
ball = img[10:20, 10:20]
img[80:90, 80:90] = ball

cv2.imshow("ball",img)
#Splitting and Merging Image Channels
b,g,r = cv2.split(img)
img = cv2.merge((b,g,r))
#or
b = img[:,:,0]
#Set all red pixel to 0
img[:,:,2] = 0

#Make border of image



BLUE = [255,0,0]

img1 = cv2.imread('opencv_logo.jpg')

replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')

plt.show()