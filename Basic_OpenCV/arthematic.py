import cv2
import numpy as np
from matplotlib import pyplot as plt
x=np.uint8([250])
y=np.uint8([10])
print x, y
print cv2.add(x,y)

#Add images or image blending

# img1 = cv2.imread('bl.jpg')
# img2 = cv2.imread('logo.jpg')

# dst = cv2.addWeighted(img1,0.7,img2,0.3,0)

# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# bit_wise images

# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_log.jpg')



# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

# Put logo in ROI and modify the main image


# plt.subplot(241),plt.imshow(img1,'gray'),plt.title('image')
# plt.subplot(242),plt.imshow(img2,'gray'),plt.title('logo')
# plt.subplot(243),plt.imshow(img2gray,'gray'),plt.title('logogray')
plt.subplot(241),plt.imshow(mask,'gray'),plt.title('mask')
plt.subplot(243),plt.imshow(mask_inv,'gray'),plt.title('mask_inv')
plt.subplot(245),plt.imshow(img1_bg,'gray'),plt.title('img1_mask bitwise_and')
plt.subplot(247),plt.imshow(img2_fg,'gray'),plt.title('img2_mask bitwise_and')
plt.show()
dst = cv2.add(img1_bg,img2_fg)
#dst = cv2.add(img1[0:rows, 0:cols],img2)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

