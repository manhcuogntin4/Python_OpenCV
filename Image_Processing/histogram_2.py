#Histogram Equalization use it to improve the contrast of our images
# Histogram stretch  histogram to either ends

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('wiki.jpg',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])
print hist
cdf = hist.cumsum()
print cdf
cdf_normalized = cdf * hist.max()/ cdf.max()
print cdf_normalized
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()