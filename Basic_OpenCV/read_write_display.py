import numpy as np
import cv2

# Read image
img = cv2.imread('messi5.jpg',0)

# Display image

cv2.imshow('image',img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
	#Save image
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()
