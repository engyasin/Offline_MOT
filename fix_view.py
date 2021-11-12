import cv2
import numpy as np
from background_substraction import BG_substractor 

from config import config
from utils import resize


class FixView():
    def __init__(self,bg_rgb):
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=2000)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 500)

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        bg = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2GRAY)

        # taking from the border to avoid, foreground
        mask =  np.zeros_like(bg,dtype=np.uint8)

        borders_ = 1200
        mask[:borders_,:] = 255
        mask[-borders_:,:] = 255
        mask[:,:borders_] = 255
        mask[:,-borders_:] = 255
        self.mask_bg = mask.copy()

        self.kps_bg,self.des_bg = self.sift.detectAndCompute(bg,self.mask_bg)

        #dilation for mask of the frames
        kernel_size = config.fixing_dilation
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
        self.min_matches = config.min_matches

    def get_good_matches(self,matches):
        good = []
        for mn in matches:
            if len(mn)==2:
                m,n=mn
                if m.distance < 0.7*n.distance:
                    good.append(m)
            else:
                #try adding anyway
                good.append(mn[0])
                pass
        # take only the best 1000
        good = sorted(good,key=lambda x: x.distance)[:500]
        return good

    def fix_view(self,frame,fgmask=None):

        if fgmask is None:
            mask = self.mask_bg.copy()
        else:
            fg_img = cv2.dilate(fgmask,self.kernel,iterations = 1)
            mask = np.zeros_like(fg_img,dtype=np.uint8)
            mask[fg_img==0] = 255

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, des = self.sift.detectAndCompute(gray,mask)

        matches = self.flann.knnMatch(self.des_bg,des,k=2)
        good = self.get_good_matches(matches)

        if len(good)>self.min_matches:
            src_pts = np.float32([ self.kps_bg[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kps[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M,dummy = cv2.estimateAffine2D(dst_pts[:,0,:],src_pts[:,0,:])
            M = np.round(M,4)
            if dummy.sum()>self.min_matches:
                frame = cv2.warpAffine(frame,M,gray.shape[::-1],flags=cv2.INTER_CUBIC,borderValue=0)
            else:
                print("Not transformed, there is low number of correcr matches")
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), self.min_matches) )

        return frame

    def set_new_bg(self,bg_rgb,fg_mask=None):
        bg = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2GRAY)
        if fg_mask is None:
            mask = self.mask_bg.copy()
        else:
            fg_img = cv2.dilate(fg_mask,self.kernel,iterations = 1)
            mask = np.zeros_like(fg_img,dtype=np.uint8)
            mask[fg_img==0] = 255
        self.kps_bg,self.des_bg = self.sift.detectAndCompute(bg,mask)


if __name__ == '__main__':

    cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')
    frame_id = 1
    cap.set(1, frame_id-1)
    ret,bg_rgb = cap.read()

    Fix_obj = FixView(bg_rgb)
    BG_s = BG_substractor(bg_rgb)

    ret, frame = cap.read()

    fg_img= BG_s.bg_substract(frame)

    while(ret):

        frame = Fix_obj.fix_view(frame,fg_img)

        fg_img = BG_s.bg_substract(frame)

        cv2.imshow('fgmask', resize(fg_img,0.2)) 
        print(frame_id)
        k = cv2.waitKey(30) & 0xff
        #prv_regions = []
        if k == 27: 
            break

        ret, frame = cap.read()
        frame_id += 1

    cap.release() 
    cv2.destroyAllWindows() 