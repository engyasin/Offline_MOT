# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element, postprocForg


def detect_firstF(Frames):
	"""
	input: group of Frames
	output: list of rectengles for objects in the first frame of input
	"""
	LEN = len(Frames)
	fgbg = cv2.createBackgroundSubtractorKNN(LEN,200,True)
	for i,frame in enumerate(Frames):
		fgmask = fgbg.apply(frame,learningRate = float((LEN-i)/LEN))

	fgmask = fgbg.apply(Frames[0],learningRate = 0.0)
	fgmask = postprocForg(fgmask)

	label_image = label(fgmask)
	regs_str = regionprops(label_image)

	rectengles = []

	STD_Thresh = 6

	for r in regs_str:
		gray_im = cv2.cvtColor(frame[r.slice], cv2.COLOR_BGR2GRAY)

		#if np.std(np.histogram(frame[r.slice],bins=256,range=(0,256))[0])<10:
		#if gray_im.std()<13:
		if np.std(np.histogram(gray_im,bins=256,range=(0,256))[0])<STD_Thresh:
			fgmask[r.slice] = 0
		else:
			X = r.centroid
			rc = (r.image.shape[0]/2,r.image.shape[1]/2)
			rectengles.append([tuple((int(X[1]-rc[1]),int(X[0]-rc[0])))
								,tuple((int(X[1]+rc[1]),int(X[0]+rc[0]))),[255,0,0]])
			#print(gray_im.std())
	return rectengles

def main():
	batch_s = 8

	cap = cv2.VideoCapture('./video.mov') 
	#fgbgmog = cv2.createBackgroundSubtractorMOG2()#10,50) 

	Frames = []

	for i in range(batch_s):
		ret,frame = cap.read()
		Frames.append(frame)

	# for the first image

	rectengles = detect_firstF(Frames)

	Toshow = Frames[0].copy()

	for rect in rectengles:
		cv2.rectangle(Toshow,rect[0],rect[1],rect[2],2)
	scale_percent = 50 # percent of original size

	I = cv2.resize(Toshow,
						tuple([int(x*scale_percent/100) for x in Toshow.shape[::-1][1:]]))

	while(1):

		cv2.imshow('fgmask', I)
		k = cv2.waitKey(30) & 0xff
		
		if k == 27: 
			break

	#cv2.imshow('frame',frame ) 
	cap.release()


if __name__ == "__main__":
	main()
