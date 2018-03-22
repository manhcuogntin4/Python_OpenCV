import cv2
import numpy as np
import glob
import imutils
import os
import sys
from string import Template
import subprocess

IMAGES_FOLDER="./"



def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list




ls_images=readFileImages(IMAGES_FOLDER)
for filename in ls_images:
	rc=subprocess.check_call(["./textcleaner", "-u", filename, filename])
