# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element, assocciate, not_within_frame, wighted_associate
import sys
from vidstab import VidStab
from draft import simplifyimg



def map_frame(img,centers):
	p = len(centers)
	distances = np.zeros((*img.shape[:-1],p))
	i = 0
	for c in centers:
		distances[:,:,i] = np.sum(abs(c-np.int16(img)),axis=2)
		i += 1
	labels = np.argmin(distances,axis=2)
	res = centers[labels.flatten()]
	return res.reshape((img.shape))

def Template_p_match(image, template , bbox,mask):
    W = int((bbox[2] -bbox[0])*1.5)
    H = int((bbox[3] -bbox[1])*1.5)
    Nbox = [ max(0,bbox[0]- W) , max(0,bbox[1]- H), bbox[2] + W, bbox[3] + H]
    #print(image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    #print(template.shape)
    #print(image.shape,bbox)
    #template[template == 0] = 127
    N_templ = image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:]
    res = cv2.matchTemplate(N_templ,template,
    cv2.TM_CCORR_NORMED,mask=np.float32(mask))
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    #print("***",max_val)
    #print(res[max_loc[::-1]],max_val)
    #print(max_loc,image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    # test the value?

    return Nbox,max_val,(max_loc[0]+Nbox[1],max_loc[0]+Nbox[0])





history = 10

cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')
#cap2= cv2.VideoCapture('../../DJI_0148.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(history,100,False)
# history, distance2threshold, detect-shadow
#fgbg = cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=10,detectShadows=True)# 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

smoothing_kernel = np.ones((9,9),np.float32)/81

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None

Final_Result = []

frame_id = 0
prv_regions = []
prv_frame = []
Thresh_move = 15

Traffic = []
ret = True
ret, frame = cap.read()
#centers,_ = simplifyimg(frame,parts=25)

#trackers = cv2.legacy_MultiTracker()

stabilizer = VidStab()

max_background = 0
Best_bg = np.zeros(0)
rate_learn = 0.5

while(ret): 
	ret, frame = cap.read()
	if not(ret) :
		break
	#frame = map_frame(frame,centers)
	#print('shape: ',frame.shape)
	#if frame_id%batch_s ==0:
	#	for i in range(batch_s):
	#		_,frame2 = cap2.read()
	#		fgmask = fgbg.apply(frame2,learningRate = float((batch_s-i)/batch_s))
	frame_id += 1
	#if frame_id%3 : continue 
	stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=5,smoothing_window=1)
	out = np.array([])

	fgmask = fgbg.apply(stabilized_frame,out,learningRate = rate_learn)
	#(success, boxes) = trackers.update(stabilized_frame)
	#print ('sucess :',success)
	#_,fgmask = cv2.threshold(fgmask,254,255,cv2.THRESH_BINARY)

	# find shadows
	#fgmask_shadows = (fgmask==127)
	#fgmask[fgmask>1] = 255
	#to bool
	#fgmask = np.array([fgmask_prev>1],dtype=bool)

	#fgmask = cv2.erode(fgmask,kernel,iterations = 1)
	#fgmask = cv2.filter2D(fgmask,-1,smoothing_kernel)
	#fgmask = cv2.medianBlur(fgmask,5)

	#fgmask[fgmask<255] = 0


    # to cut object off
	#label_image = label(fgmask)

	#Accepted_ratio = 0.25#changed
	#Accepted_extent = 0.38#changed

	# first 4 frames are wrong
	#regs_str = regionprops(label_image)

	#Traffic = Tr[:]
	#fgmask[rgn.slice] = rgn.image

	#label_image = label(fgmask)
	#prv_regions =  regionprops(label_image,frame)
	label_image = label(fgmask)
	regs_str = regionprops(label_image,stabilized_frame)
	#if frame_id>20:
	#	breakpoint()
	#breakpoint()
	for r in regs_str:
		if r.area < 400:
			fgmask[r.slice] = 0
		elif r.extent < 0.1:
			fgmask[r.slice] = 0
		else:
			#continue
			#prv_regions.append(r.image.copy())
			dims = (r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0])
            #print(dims)
			rect = cv2.boxPoints((r.centroid[::-1],dims,-1*np.rad2deg(r.orientation)))
			rect = np.intp(rect)
            #cv2.drawContours(frame, [rect], 0, detection.color,-1)
			#cv2.drawContours(stabilized_frame, [rect], 0, (255,0,0),4)
			cv2.rectangle(stabilized_frame,r.bbox[1::-1],r.bbox[3:1:-1],color=(255,0,0),thickness=4)

	N = np.sum(fgmask)
	if (max_background< N) and (frame_id>10) and (frame_id < 220):
		max_background = N
		# take that view
		Best_bg = fgbg.getBackgroundImage()
		rate_learn = 0.1
	#print(rate_learn)

	if frame_id == -10:
		for r in regs_str:
			if r.area>450:
				box = [r.bbox[1],r.bbox[0],r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0]]
				object_tracker = cv2.legacy.TrackerMIL_create()
				#trackers.add(object_tracker, stabilized_frame, box)
	fgmask_3 = np.dstack((fgmask,fgmask,fgmask))
	#Toshow = frame.copy()
	#prv_frame = Toshow.copy()
	#for rect in rectengles:
		#print(rect)
		#cv2.rectangle(Toshow,rect[0],rect[1],rect[2],2)
	#for box in boxes:
	#	(x, y, w, h) = [int(v) for v in box]
	#	cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

	I_com = np.hstack((fgmask_3,stabilized_frame))#fgmask_3,fgbg.getBackgroundImage(

	scale_percent = 20 # percent of original size

	I_com = cv2.resize(I_com,
						tuple([int(x*scale_percent/100) for x in I_com.shape[::-1][1:]]))
	#plt.imshow(cv2.cvtColor(I_com, cv2.COLOR_BGR2RGB))
	#if writer is None:
	#	writer = cv2.VideoWriter("output.avi", fourcc, 30,
	#	(I_com.shape[1], I_com.shape[0]), True)
	#if writer is None:
	#	writer = cv2.VideoWriter("output.avi", fourcc, 30,
	#	(I_com.shape[1], I_com.shape[0]), True)
	#writer.write(I_com)
	cv2.imshow('fgmask', I_com) 
	#cv2.imshow('frame',frame ) 



	k = cv2.waitKey(10) & 0xff
	#prv_regions = []
	if k == 27: 
		break

cap.release() 
#cap2.release()
#writer.release()
np.save('background.png',Best_bg)
cv2.destroyAllWindows() 

