# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element, assocciate, not_within_frame, wighted_associate



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


def get_quard(gp1,gp2,dims):
	H,W = dims # y,x
	new_gp1,new_gp2 = [],[]
	points = [[0,0],[W,0],[W,H],[0,H]]
	for p in points:
		indx = np.argmin([np.linalg.norm(x) for x in np.array(p)-gp1])
		new_gp1.append(gp1[indx])
		gp1 = np.delete(gp1,indx,0)
		new_gp2.append(gp2[indx])
		gp2 = np.delete(gp2,indx,0)
	return np.array(new_gp1),np.array(new_gp2)


batch_s = 5

cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')
#cap2= cv2.VideoCapture('../../DJI_0148.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(batch_s,100,False)
# history, distance2threshold, detect-shadow
#fgbgmog = cv2.createBackgroundSubtractorMOG2()#10,50) 
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

trackers = cv2.legacy_MultiTracker()

#stabilizer = VidStab()


bg = np.load('background.png.npy')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()

#ORB_detector = cv2.ORB_create(nfeatures=2000)

# FLANN parameters (ORB)
FLANN_INDEX_LSH = 6
index_params1= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params1 = dict(checks=1000)   # or pass empty dictionary

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 500)


flann = cv2.FlannBasedMatcher(index_params,search_params)


mask =  np.zeros_like(bg,dtype=np.uint8)

mask[:500,:] = 255
mask[-500:,:] = 255
mask[:,:600] = 255
mask[:,-600:] = 255

##
#kps_bg,des_bg = ORB_detector.detectAndCompute(bg,mask)
kps_bg,des_bg = sift.detectAndCompute(bg,mask)

max_background = 0
rate_learn = 0.5

MIN_MATCH_COUNT = 5

while(ret):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#kps, des = ORB_detector.detectAndCompute(gray,mask)
	kps, des = sift.detectAndCompute(gray,mask)

	matches = flann.knnMatch(des_bg,des,k=2)
	#matches = [m for m in matches if len(m)==2]
	#frame = map_frame(frame,centers)
	#print('shape: ',frame.shape)
	#if frame_id%batch_s ==0:
	#	for i in range(batch_s):
	#		_,frame2 = cap2.read()
	#		fgmask = fgbg.apply(frame2,learningRate = float((batch_s-i)/batch_s))
	# store all the good matches as per Lowe's ratio test.
	#print(matches)
	good = []
	for mn in matches:
		if len(mn)==2:
			m,n=mn
			if m.distance < 0.7*n.distance:
				good.append(m)
	# take only the best 10
	good = sorted(good,key=lambda x: x.distance)[:200]
	if len(good)>MIN_MATCH_COUNT:
		src_pts = np.float32([ kps_bg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kps[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# find the square:
		#src_pts,dst_pts = get_quard(src_pts,dst_pts,gray.shape[:2])
		#breakpoint()
		#M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		#M = cv2.getPerspectiveTransform(dst_pts[:4,0,:],src_pts[:4,0,:],solveMethod=cv2.DECOMP_CHOLESKY)
		M,_ = cv2.estimateAffine2D(dst_pts[:,0,:],src_pts[:,0,:])#,solveMethod=cv2.DECOMP_CHOLESKY)
		print( dst_pts[:3,0,:])
		print(src_pts[:3,0,:])
		#matchesMask = mask.ravel().tolist()
		print('M: ',M)
		frame = cv2.warpAffine(frame,M,gray.shape[::-1],borderValue=10)

		#pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		#dst = cv2.perspectiveTransform(pts,M)
		#img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
	else:
		print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
		matchesMask = None



	frame_id += 1
	#if frame_id%3 : continue 

	#stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=5,smoothing_window=1)
	out = np.array([])

	fgmask = fgbg.apply(frame,out,learningRate = rate_learn)
	#(success, boxes) = trackers.update(frame)
	#print ('sucess :',success)

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
	regs_str = regionprops(label_image,frame)
	#if frame_id>20:
	#	breakpoint()
	#breakpoint()
	for r in regs_str:
		if r.area < 500:
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
			cv2.rectangle(frame,r.bbox[1::-1],r.bbox[3:1:-1],color=(255,0,0),thickness=4)

	N = np.sum(fgmask)
	if (max_background< N) and (frame_id>10):
		max_background = N
		# take that view
		rate_learn = 0
	#print(rate_learn)

	if frame_id == -10:
		for r in regs_str:
			if r.area>450:
				box = [r.bbox[1],r.bbox[0],r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0]]
				object_tracker = cv2.legacy.TrackerMIL_create()
				trackers.add(object_tracker, frame, box)
	fgmask_3 = np.dstack((fgmask,fgmask,fgmask))
	#Toshow = frame.copy()
	#prv_frame = Toshow.copy()
	#for rect in rectengles:
		#print(rect)
		#cv2.rectangle(Toshow,rect[0],rect[1],rect[2],2)
	#for box in boxes:
	#	(x, y, w, h) = [int(v) for v in box]
	#	cv2.rectangle(stabilized_frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

	I_com = np.hstack((fgmask_3,frame))#fgmask_3,fgbg.getBackgroundImage(

	scale_percent = 20 # percent of original size

	I_com = cv2.resize(I_com,
						tuple([int(x*scale_percent/100) for x in I_com.shape[::-1][1:]]))
	#plt.imshow(cv2.cvtColor(I_com, cv2.COLOR_BGR2RGB))
	#if writer is None:
	#	writer = cv2.VideoWriter("output.avi", fourcc, 30,
	#	(I_com.shape[1], I_com.shape[0]), True)
	if writer is None:
		writer = cv2.VideoWriter("output1.avi", fourcc, 30,
		(I_com.shape[1], I_com.shape[0]), True)
	writer.write(I_com)
	cv2.imshow('fgmask', I_com) 
	#cv2.imshow('frame',frame ) 



	k = cv2.waitKey(10) & 0xff
	#prv_regions = []
	if k == 27: 
		break

cap.release() 
#cap2.release()
writer.release()

cv2.destroyAllWindows() 

