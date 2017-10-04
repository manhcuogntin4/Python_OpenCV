import cv2
import numpy as np
#
# ------------ Main
def eraseContour(img, y, x, h, w):
	xPos=x
	yPos=y
	while xPos <= x+w: #Loop through rows
		while yPos <= y+h:
			img.itemset((yPos, xPos, 0), 255) 
			img.itemset((yPos, xPos, 1), 255)
			img.itemset((yPos, xPos, 2), 255)  
			yPos = yPos + 1
		yPos = y
		xPos = xPos + 1 #Increment X position by 1	


def cleanImage(imgPath)	
	image = cv2.imread(imgPath, 1)
	if image.shape[2]>1:
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
	#gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
	_,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV) # threshold
	kernel = cv2.getStructuringElement(cv2.MORPH_OPEN,(3,3))
	dilated = cv2.dilate(thresh,kernel,iterations = 2) # dilate
	_,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
	mask = np.ones(image.shape[:2], dtype="uint8") * 255	
	height, width, channels = image.shape
	for contour in contours:
	# get rectangle bounding contour
		area = cv2.contourArea(contour)
		[x,y,w,h]=cv2.boundingRect(contour)   
		#if ((area < 400 and ( y< 0.2*height)) or (area < 400 and ( y> 0.8*height))):
		if ((h < height/4 and ( y< 0.3*height)) or (h < height/4 and ( y> 0.7*height))):
			eraseSmallContour(image,y,x,h,w)
	outPath='out.png'	
	cv2.imwrite(outPath, image)
	
