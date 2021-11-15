# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 
from skimage.measure import label, regionprops
import sys
from utils import resize

from config import config

class BG_substractor():
    def __init__(self,bg):

        self.history = config.bgs_history
        self.thresh  = config.bgs_threshold
        self.shadows = config.bgs_shadows
        self.fgbg = cv2.createBackgroundSubtractorKNN(self.history,self.thresh,self.shadows)
        self.bg = bg.copy()

        kernel_size = config.bgs_erosion_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))

    def bg_substract(self,frame):

        # bg is the old background
        # frame is the image
        #TODO use class methods instead
        out = np.array([])
        frame_rate ,bg_rate = 0.3, 0.5
        for _ in range(4):
            # 4 times to learn bg more
            fgmask = self.fgbg.apply(self.bg,out,learningRate = bg_rate)

        fgmask = self.fgbg.apply(frame,out,learningRate = frame_rate)

        #_,fgmask = cv2.threshold(fgmask,254,255,cv2.THRESH_BINARY)

        fgmask = cv2.erode(fgmask,self.kernel,iterations = 1)
        #fgmask = cv2.filter2D(fgmask,-1,smoothing_kernel)
        #fgmask = cv2.medianBlur(fgmask,5)
        #fgmask[fgmask<255] = 0

        self.bg = self.fgbg.getBackgroundImage()

        return fgmask

    def get_big_objects(self,fg_mask,frame):

        label_image = label(fg_mask)
        regs_str = regionprops(label_image,frame)
        new_regions = []
        for r in regs_str:
            if r.area < config.bgs_min_area:
                fg_mask[r.slice] = 0
            elif r.extent < 0.1:
                fg_mask[r.slice] = 0  
            else:
                new_regions.append(r)

        return fg_mask,new_regions



if __name__ == '__main__':

    cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')
    ret, bg = cap.read()
    frame_id = 1
    cap.set(1, frame_id-1)
    BG_s = BG_substractor(bg)
    ret, frame = cap.read()
    while ret: 
        frame_id += 1
        I_com = BG_s.bg_substract(frame)
        cv2.imshow('fgmask', resize(I_com,0.2)) 
        print(frame_id)
        k = cv2.waitKey(30) & 0xff
        #prv_regions = []
        if k == 27: 
            break
        ret, frame = cap.read()

    cap.release() 
    cv2.destroyAllWindows() 