# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element, assocciate, not_within_frame, wighted_associate

batch_s = 10

cap = cv2.VideoCapture('../../DJI_0148.mp4') 
cap2= cv2.VideoCapture('../../DJI_0148.mp4')
fgbg = cv2.createBackgroundSubtractorKNN(batch_s,40,True)
# history, distance2threshold, detect-shadow
#fgbgmog = cv2.createBackgroundSubtractorMOG2()#10,50) 
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))


fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = None

Final_Result = []

frame_id = 0
prv_regions = []
prv_frame = []
Thresh_move = 15

Traffic = []

while(1): 
	ret, frame = cap.read()
	print('shape: ',frame.shape)
	if frame_id%batch_s ==0:
		for i in range(batch_s):
			_,frame2 = cap2.read()
			fgmask = fgbg.apply(frame2,learningRate = float((batch_s-i)/batch_s))
	frame_id += 1
	#if frame_id%3 : continue 
	fgmask = fgbg.apply(frame,learningRate = 0.1)

	# find shadows
	fgmask_shadows = (fgmask==127)
	fgmask[fgmask>1] = 255

	fgmask = cv2.erode(fgmask,kernel,iterations = 1)

    # to cut object off
	label_image = label(fgmask)

	Accepted_ratio = 0.25#changed
	Accepted_extent = 0.38#changed

	# first 4 frames are wrong
	regs_str = regionprops(label_image)
	for region in regs_str:
		a = region.major_axis_length 
		a+= 0.00001
		b = region.minor_axis_length
		shadow_r = 1
		if np.any(fgmask_shadows[label_image==region.label]):
			shadow_r = 0.7
		if((b/a)<(Accepted_ratio*shadow_r)) or region.extent<(Accepted_extent*shadow_r):
			# not an object (line maybe)
			fgmask[region.slice] = 0

	fgmask = cv2.dilate(fgmask,kernel_dilate,iterations = 2)

	label_image = label(fgmask)
	regs_str = regionprops(label_image)

	# another filter for objects
	for r in regs_str:
		if np.std(np.histogram(frame[r.slice],bins=256,range=(0,256))[0])<7:
			fgmask[r.slice] = 0


	label_image = label(fgmask)
	regs_str = regionprops(label_image)

	regs_ctrs = [r.centroid for r in regs_str]
	regs_sizes = [r.bbox_area for r in regs_str]
	rectengles = []

	prv_centers = [r.centroid for r in prv_regions]
	if frame_id == 6:
		R,dists = assocciate(prv_centers,regs_ctrs)
		for p_pairs,d  in zip(R,dists):

			rgn = prv_regions[p_pairs[0]]
			rc = rgn.image.shape
			R_ele = regs_str[p_pairs[1]]

			E = Element(frame_id-1,prv_frame[rgn.slice],rgn)
			E.add_match(frame_id,R_ele,d)
			Traffic.append(E)
			X = np.array(R_ele.centroid).astype(int)

			rectengles.append([tuple((X[1]-rc[1],X[0]-rc[0]))
			,tuple((X[1]+rc[1],X[0]+rc[0])),E.color])
	elif frame_id>6:
		mis_matched = []
		prv_centers = [r.matching_place[-1] for r in Traffic]
		prv_sizes = [np.prod(r.size[-1]) for r in Traffic]
		#R,dists = wighted_associate(prv_centers,regs_ctrs,prv_sizes,regs_sizes)
		R,dists = assocciate(prv_centers,regs_ctrs)
		for p_pairs,d  in zip(R,dists):
			rgn = Traffic[p_pairs[0]]
			#Thresh_val = [2*Thresh_move,Thresh_move][rgn.success[-1]]
			if d> rgn.get_thresh(): 
				print("***** ",d)
				mis_matched.append(rgn)
				rgn.add_mismatch(frame_id)
				if len(rgn.success)>4 and np.any(rgn.success[-6:]):
					Traffic[p_pairs[0]] = []
					# TODO save if good box
					if len(rgn.success)>24:
						Final_Result.append(rgn)
					print("some boxes deleted forever")
				else:
					Traffic[p_pairs[0]] = rgn
				continue

			rc = rgn.size[-1]
			R_ele = regs_str[p_pairs[1]]

			rgn.add_match(frame_id,R_ele,d)
			Traffic[p_pairs[0]] = rgn
			X = np.array(R_ele.centroid).astype(int)

			regs_str[p_pairs[1]] = []
			rectengles.append([tuple((X[1]-rc[1],X[0]-rc[0]))
			,tuple((X[1]+rc[1],X[0]+rc[0])),rgn.color])
	Margin = 10
	for Rgn in regs_str:
		if not(Rgn):continue
		T = not_within_frame(frame,Rgn.centroid,Margin)
		if frame_id>6 and T:
			# is it comming? TODO
			#if Rgn.centroid[0] <30 or Rgn.centroid[1]
			E = Element(frame_id,frame[Rgn.slice],Rgn, matched=True)
			Traffic.append(E)
		elif not(T):
			print("shape ",Rgn.image.shape)
	

	#fgmask = cv2.dilate(fgmask,kernel_dilate,iterations = 3)
	Tr = []
	for entity in Traffic[:]:
		if entity:
			if not(not_within_frame(frame,entity.matching_place[-1],6)):
				Tr.append(entity)
			else:
				#TODO save if good
				if len(entity.success)>19:
					Final_Result.append(entity)
				print("+++++",entity.matching_place[-1])
	Traffic = Tr[:]
	#fgmask[rgn.slice] = rgn.image

	label_image = label(fgmask)
	prv_regions =  regionprops(label_image,frame)

	fgmask_3 = np.dstack((fgmask,fgmask,fgmask))
	Toshow = frame.copy()
	prv_frame = Toshow.copy()
	for rect in rectengles:
		#print(rect)
		cv2.rectangle(Toshow,rect[0],rect[1],rect[2],2)

	I_com = np.hstack((fgmask_3,Toshow))#fgmask_3,fgbg.getBackgroundImage(

	scale_percent = 50 # percent of original size

	I_com = cv2.resize(I_com,
						tuple([int(x*scale_percent/100) for x in I_com.shape[::-1][1:]]))
	#plt.imshow(cv2.cvtColor(I_com, cv2.COLOR_BGR2RGB))
	if writer is None:
		writer = cv2.VideoWriter("output.avi", fourcc, 30,
		(I_com.shape[1], I_com.shape[0]), True)
	writer.write(I_com)
	cv2.imshow('fgmask', I_com) 
	#cv2.imshow('frame',frame ) 

	k = cv2.waitKey(30) & 0xff
	
	if k == 27: 
		break

cap.release() 
cap2.release()
writer.release()
cv2.destroyAllWindows() 


place_h = np.array([[-1,-1,-1,-1,-1] for _ in range(frame_id)])
Towrite = place_h.ravel().copy()
for E in Final_Result:
	L = place_h.copy()
	for i,f in enumerate(E.frame_ids):
		# centroid , size, success
		L[f-1] = E.matching_place[i][0],E.matching_place[i][1],\
		E.size[i][0],E.size[i][1],E.success[i]
	Towrite = np.vstack((Towrite,L.ravel()))

np.savetxt("Trajectories.csv",Towrite[1:],delimiter=",")

print(len(Final_Result))
