import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import sys, getopt
import os
import difflib
import sys

#Process parallel
from multiprocessing import Pool   
import multiprocessing 
import subprocess
from multiprocessing import Manager
from functools import partial
import multiprocessing.pool

import pytesseract
from PIL import Image

import glob
from pandas import DataFrame

import subprocess
import argparse

#IMAGES_FOLDER="./"

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
	help="images folder")

ap.add_argument("-c", "--class", required=True,
	help="model path")

ap.add_argument("-o", "--output", required=True,
	help="excel output")

args = vars(ap.parse_args())


IMAGES_FOLDER=args["folder"]
CLASS=args["class"]
excel_outfile=args["output"]


def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list




ls_images=readFileImages(IMAGES_FOLDER)


reload(sys)
sys.setdefaultencoding('utf-8')
CACHE_FOLDER = '/tmp/caffe_demos_uploads/cache'
this_dir = os.path.dirname(__file__)

def get_similar(str_verify, isLieu=False, score=0.6):
	words = []
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	if(isLieu):
		lieu_path=os.path.join(this_dir, 'lieu.txt')	
	else:
		lieu_path=os.path.join(this_dir, 'nom.txt')			
	f=open(lieu_path,'r')
	for line in f:
		words.append(line)
	f.close()
	print str_verify
	index = 0
	if(str_verify.find(u':') != -1 and str_verify.index(u':') < 5):
		index = str_verify.index(u':')+1

	str_verify=str_verify[index:]
	print str_verify
	simi=difflib.get_close_matches(str_verify, words,5,score)
	#print(simi)
	return simi

def convert_to_binary(img):
	cv2.setNumThreads(0)
	if (img.shape >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	res = cv2.fastNlMeansDenoising(res,20, 7, 21)
	out_path = os.path.join(CACHE_FOLDER, str(os.getpid())+ "out.png")
	cv2.imwrite(out_path,res)
	return out_path, res



def extract_text(img_path, model_path):
	ocr = pyclstm.ClstmOcr()
	ocr.load(model_path)
	imgFile = Image.open(img_path)
	text = ocr.recognize(imgFile)
	text.encode('utf-8')
	chars = ocr.recognize_chars(imgFile)
	prob = 1
	index = 0
	#print text
	if(text.find(u':') != -1 and text.index(u':') < 5):
		index = text.index(u':')+1
	if(text.find(u' ') != -1 and (text.index(u' ') <= 3)):
		if(len(text)>text.index(u' ')+1):		
			index = text.index(u' ')+1
	for ind, j in enumerate(chars):
		#print j
		if ind >= index:		
			prob *= j.confidence	
	#return text[index:], prob, index
	return text, prob, index


def crop_image(img,cropX=0, cropY=0, cropWidth=0, cropHeight=0):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	if(h-cropHeight>cropY) and (w-cropWidth>cropX):
		res=img[cropY:h-cropHeight, cropX:w-cropWidth]
	else:
		res=img[cropY:h,cropX:w]
	print str(os.getpid())
	out_path = os.path.join(CACHE_FOLDER, str(os.getpid())+"croped.png")
	#out_path = os.path.join(CACHE_FOLDER, "croped.png")
	cv2.imwrite(out_path,res)
	return out_path


def clstm_ocr(img, cls):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if cls=="lieu":
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')

	islieu=False
	if cls=="lieu":
		islieu=True

	print model_path
	converted_image_path, image = convert_to_binary(img)
	#maxPro = 0
	#ocr_result = ""
	ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
	if(index>0):
		image=image[10:,:]
	cropX=1
	cropY=8
	cropWidth=1
	cropHeight=10
	if(islieu):
		cropHeight=3
		cropY=3
		cropX=3
		cropWidth=3
	for i in range (0,cropX,1):
		for j in range (0,cropY):
			for k in range (0,cropWidth):
				for h in range (0, cropHeight):
					img_path = crop_image(image, i, j, k, h)
					text, prob, index = extract_text(img_path, model_path)
					#print text, prob
					if(prob > maxPro) and (len(text)>=2):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2):
						break	
	#print maxPro, ocr_result	
	if(islieu and maxPro<1):	
		if(maxPro<0.9):
			ocr=get_similar(ocr_result,islieu,0.5)
			if(len(ocr)>0):
				ocr_result=ocr[0].encode('utf-8')+" : "+ ocr_result
		else:
			ocr=get_similar(ocr_result,islieu,0.6)
			if(len(ocr)>0):
				ocr_result=ocr[0].encode('utf-8') +" : "+ ocr_result
	return (ocr_result, maxPro)


def clstm_ocr_parallel(img, cls):
	try:
		os.mkdir(CACHE_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(CACHE_FOLDER, 0755)

	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if cls=="lieu":
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')

	print model_path

	if(img.size>0):
		print "size >0"
		converted_image_path, image = convert_to_binary(img)
		ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
		print "first extract finished"
	else:
		return ("",0)
	#maxPro = 0
	#ocr_result = ""
	
	if(index>0):
		image=image[10:,:]
	cropX=1
	cropY=8
	cropWidth=1
	cropHeight=10
	islieu=False
	if (cls=="lieu"):
		islieu=True
	if(islieu):
		cropHeight=3
		cropY=3
		cropX=3
		cropWidth=3
	q={}
	p={}
	txt={}
	prob={}
	for j in range (0,cropY):
		q[j] = multiprocessing.Queue()
		p[j] = multiprocessing.Process(target=calib_clstm_height_low_queue, args=(cropX,j,cropWidth,cropHeight,image,model_path,q[j]))
		p[j].start()
	for j in range (0,cropY):
		#p[j].join()
		txt[j],prob[j]=q[j].get()
	for j in range (0,cropY):
		p[j].join()
		#txt[j],prob[j]=q[j].get()
	for j in range (0,cropY):
		if(prob[j]>maxPro):
			maxPro=prob[j]
			ocr_result=txt[j]
	#print "Here:", os.getpid()

	#print maxPro, ocr_result	
	if(islieu and maxPro<1):	
		if(maxPro<0.9):
			ocr=get_similar(ocr_result,islieu,0.5)
			if(len(ocr)>0):
				ocr_result=ocr[0].encode('utf-8')+" : "+ ocr_result
		else:
			ocr=get_similar(ocr_result,islieu,0.6)
			if(len(ocr)>0):
				ocr_result=ocr[0].encode('utf-8') +" : "+ ocr_result
	return (ocr_result, maxPro)



def clstm_ocr_calib(img, cls):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model_nomprenom_cni.clstm')
	if cls=="lieu":
		model_path = os.path.join(this_dir, 'model_lieu_cni.clstm')
	if cls=="datenaissance":
		model_path = os.path.join(this_dir, 'model_date_cni.clstm')
	if cls=="mrz1" or cls=="mrz2":
		model_path = os.path.join(this_dir, 'model_mrz_cni_new.clstm')
	ocr_result, maxPro="",0
	islieu=False
	if cls=="lieu":
		islieu=True
	if(img.size>0):

		converted_image_path, image = convert_to_binary(img)
		#maxPro = 0
		#ocr_result = ""
		ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
	
		if(islieu):	
			if(maxPro<0.9):
				ocr=get_similar(ocr_result,islieu,0.5)
				if(len(ocr)>0):
					ocr_result=ocr[0].encode('utf-8')+" : "+ ocr_result
			else:
				ocr=get_similar(ocr_result,islieu,0.6)
				if(len(ocr)>0):
					ocr_result=ocr[0].encode('utf-8')+" : "+ ocr_result
	else:
		return ("",0)
	return (ocr_result, maxPro)

def calib_clstm_height_low(cropX, y, cropWidth, cropHeight, image, model_path):
	maxPro=0
	ocr_result=""

	for i in range (0,cropX):
			for k in range (0,cropWidth):
				for h in range (0, cropHeight):
					img_path = crop_image(image, i, y, k, h)
					text, prob, index = extract_text(img_path, model_path)
					os.remove(img_path)
					if(prob > maxPro) and (len(text)>=2):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2):
						break
	return ocr_result, maxPro	

def calib_clstm_height_low_queue(cropX, y, cropWidth, cropHeight, image, model_path, q):
	q.put(calib_clstm_height_low(cropX, y, cropWidth, cropHeight, image, model_path))

def crop_image_cls(img, class_name, cropX=0, cropY=0, cropWidth=0, cropHeight=0):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	bbox=0
	if class_name == 'nom':
		bbox += 1.5 * h
	elif class_name == 'nomepouse':
		bbox += 2.5 * h
	elif class_name == 'prenom':
		bbox += 2.5 * h
	elif class_name == 'lieu':
		bbox += 0.8 * h
	if(h-cropHeight>cropY) and (w-cropWidth>cropX):
		res=img[cropY:h-cropHeight, bbox:w-cropWidth]
	else:
		res=img[cropY:h,cropX:w]
	print str(os.getpid())
	print bbox
	out_path = os.path.join(CACHE_FOLDER, str(os.getpid())+"croped.png")
	#out_path = os.path.join(CACHE_FOLDER, "croped.png")
	cv2.imwrite(out_path,res)
	return out_path



ls_txt, ls_prob=[],[]

print ls_images
for filename in ls_images:
	txt, prob= pytesseract.image_to_string(Image.open(filename)), 1
	print txt
	ls_txt.append(txt)
	ls_prob.append(prob)


df=DataFrame({'File ID':ls_images, CLASS: ls_txt, 'Prob': ls_prob})
df.to_excel(excel_outfile, sheet_name='Match_ID_File', index=False)



