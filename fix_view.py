
import cv2
import numpy as np

def fix_view(frame,bg):
    pass


bg = np.load('background.png.npy')
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.SIFT_create()

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
cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')

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