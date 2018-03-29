# Graph
# plot with pixel values (ranging from 0 to 255, not always) in X-axis
#corresponding number of pixels in the image on Y-axis
#BINS histSize : a range of value. 
#RANGE : It is the range of intensity values you want to measure.
# Normally, it is [0,256]
#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
#images : it is the source image
#channels : it is also given in square brackets. # It is the index of channel
#mask : mask image.
#histSize : this represents our BIN count.
#ranges : this is our RANGE, Normally, it is [0,256].
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('home.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])

print hist
plt.plot(hist)
plt.ylabel('Histo')
plt.xlabel('intensity')
plt.show()

# Draw 3 channels

img = cv2.imread('home.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[50:150, 40:160] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()




