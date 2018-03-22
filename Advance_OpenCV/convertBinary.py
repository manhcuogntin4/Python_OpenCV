import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('test0.png',1)
if (img.shape>=3):
	img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Otsu's thresholding
ret, img1 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite("out.png",img1)
cv2.waitKey(0)
