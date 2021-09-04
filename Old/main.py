# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element
#from detection_funcs import detect_frnn
#from calssification import classifyVGG

cap = cv2.VideoCapture('./video.mov') 
fgbg = cv2.createBackgroundSubtractorKNN(10,500,True) 
#fgbgmog = cv2.createBackgroundSubtractorMOG2()#10,50) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

frame_id = 0
prv_regions = []
prv_frame = []

while(1): 
	ret, frame = cap.read() 
	frame_id += 1
	fgmask = fgbg.apply(frame,learningRate = 0.99)

	# find shadows
	fgmask_shadows = (fgmask==127)
	fgmask[fgmask>1] = 255

	fgmask = cv2.erode(fgmask,kernel,iterations = 1)

	label_image = label(fgmask)

	Accepted_ratio = 0.25
	Accepted_extent = 0.38

	# first 4 frames are wrong
	regs_str = regionprops(label_image)
	regs_ctrs = []
	for region in regs_str:
		regs_ctrs.append(region.centroid)
		a = region.major_axis_length 
		#if a==0: continue
		a+= 0.00001
		b = region.minor_axis_length
		shadow_r = 1
		if np.any(fgmask_shadows[label_image==region.label]):
			shadow_r = 0.7
		if((b/a)<(Accepted_ratio*shadow_r)) or region.extent<(Accepted_extent*shadow_r):
			# not an object (line maybe)
			fgmask[label_image==region.label] = 0


	#detect
	rectengles = []
	if frame_id > 5:
		ToAddObjs = []
		for rgn in prv_regions:
			Nbox,scr,res = Template_p_match(frame,prv_frame[rgn.slice],rgn.bbox,rgn.image)
			rc = rgn.image.shape
			a,b = rc[0]//2,rc[1]//2
			condi = np.sum(fgmask[res[1]-a:res[1]+a+1,res[0]-b:res[0]+b+1]==255)/(rc[0]*rc[1])
			#print(np.sum(fgmask[res[1]-a:res[1]+a+1,res[0]-b:res[0]+b+1]==255),rc[0]*rc[1])
			if condi >0.99:
			#if np.any(fgmask[res[1]-rad:res[1]+rad,res[0]-rad:res[0]+rad]):
				rectengles.append([(res[0]-rc[1],res[1]-rc[0]),(res[0]+rc[1],res[1]+rc[0])])
			elif condi:
				#correction step
				#print(condi)
				A = label_image[res[1]-a:res[1]+a+1,res[0]-b:res[0]+b+1]
				#print(np.bincount(A.reshape(1,-1)[0]))
				#breakpoint()
				L = np.argmax(np.bincount(A.reshape(1,-1)[0])[1:])+1
				try: R_ind = [r.label for r in regs_str].index(L)
				except: 
					ToAddObjs.append((rgn,res))
					continue
				R_ele = regs_str.pop(R_ind)
				_ = regs_ctrs.pop(R_ind)
				X = np.array(R_ele.centroid).astype(int)
				s=[1]#_,s,c = detect_frnn(prv_frame[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:])
				#classifyVGG(prv_frame[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:])
				if np.any(s):
					#print(c)
					rectengles.append([tuple((X[1]-rc[1],X[0]-rc[0]))
					,tuple((X[1]+rc[1],X[0]+rc[0]))])
			else:
				#noise?
				# print thresh
				#print(scr)
				s=[1]#_,s,c = detect_frnn(prv_frame[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:])
				#classifyVGG(prv_frame[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:])
				if scr>0.5 and np.any(s):
					# if trusted put in the image and save it as is and draw in the matching
					# maybe we arrange acordding to size
					#rectengles.append([(res[0]-rc[1],res[1]-rc[0]),(res[0]+rc[1],res[1]+rc[0])])
					ToAddObjs.append((rgn,res))

				#breakpoint()
			#plt.imshow(cv2.cvtColor(rgn.intensity_image,cv2.COLOR_BGR2RGB))
			#plt.show()
			#cv2.imshow('d',rgn.intensity_image)
			#breakpoint()
		fgmask = cv2.dilate(fgmask,kernel_dilate,iterations = 3)
		for rgn,res in ToAddObjs:
			L = np.linalg.norm(np.array(res)-np.array(regs_ctrs),axis=1)
			rc = rgn.image.shape

			if min(L)<(1.5*max(rc)):
				print(min(L))
				R_ind = np.argmin(L)
				R_ele = regs_str.pop(R_ind)
				_ = regs_ctrs.pop(R_ind)
				X = np.array(R_ele.centroid).astype(int)

				rectengles.append([tuple((X[1]-rc[1],X[0]-rc[0]))
				,tuple((X[1]+rc[1],X[0]+rc[0]))])
			#fgmask[rgn.slice] = rgn.image
		label_image = label(fgmask)
		prv_regions =  regionprops(label_image,frame)





	fgmask_3 = np.dstack((fgmask,fgmask,fgmask))
	#fgmask_32 = np.dstack((fgmask2,fgmask2,fgmask2))
	Toshow = frame.copy()
	prv_frame = Toshow.copy()
	for rect in rectengles:
		#print(rect)
		cv2.rectangle(Toshow,rect[0],rect[1],(255,0,0),1)

	I_com = np.hstack((fgmask_3,Toshow))

	scale_percent = 50 # percent of original size

	I_com = cv2.resize(I_com,
						tuple([int(x*scale_percent/100) for x in I_com.shape[::-1][1:]]))
	#plt.imshow(cv2.cvtColor(I_com, cv2.COLOR_BGR2RGB))
	cv2.imshow('fgmask', I_com) 
	#cv2.imshow('frame',frame ) 

	#plt.show()
	k = cv2.waitKey(30) & 0xff
	
	if k == 27: 
		break
	

cap.release() 
cv2.destroyAllWindows() 
