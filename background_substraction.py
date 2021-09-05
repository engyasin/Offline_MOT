# Python code for Background subtraction using OpenCV 
import numpy as np 
import matplotlib.pyplot as plt
import cv2 
from skimage.measure import label, regionprops
from aux_func import Template_p_match, Element, assocciate, not_within_frame, wighted_associate
import sys
from utils import resize

from config import config
#from vidstab import VidStab
#from draft import simplifyimg

def bg_substract(frame,bg,fgbg_obj=None):

    # bg is the old background
    # frame is the image
    #TODO get rid of bg input
    #TODO use class methods instead
    history,thresh,shadows = config.bgs_history,config.bgs_threshold,config.bgs_shadows
    if fgbg_obj is None:
        fgbg = cv2.createBackgroundSubtractorKNN(history,thresh,shadows)
    else:
        fgbg = fgbg_obj
    out = np.array([])
    for img,rate_learn in zip([bg,bg,bg,bg,frame],[0.7,0.5,0.4,0.2,0.5]):
        fgmask = fgbg.apply(img,out,learningRate = rate_learn)

    #_,fgmask = cv2.threshold(fgmask,254,255,cv2.THRESH_BINARY)
    kernel_size = config.bgs_erosion_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    fgmask = cv2.erode(fgmask,kernel,iterations = 1)
    #fgmask = cv2.filter2D(fgmask,-1,smoothing_kernel)
    #fgmask = cv2.medianBlur(fgmask,5)
    #fgmask[fgmask<255] = 0

    """
    label_image = label(fgmask)
    regs_str = regionprops(label_image,frame)
    for r in regs_str:
        if r.area < 400:
            fgmask[r.slice] = 0
        elif r.extent < 0.1:
            fgmask[r.slice] = 0   
    """


    #N = np.sum(fgmask)
    #Best_bg = fgbg.getBackgroundImage()

    return fgmask,fgbg.getBackgroundImage(),fgbg



if __name__ == '__main__':
    pass
    cap = cv2.VideoCapture('../../DJI_0148.mp4')#Dataset_Drone/DJI_0134.mp4')
    ret, bg = cap.read()
    frame_id = 0
    fgbg_obj = None
    ret, frame = cap.read()
    while ret: 
        frame_id += 1
        I_com,bg,fgbg_obj = bg_substract(frame,bg,fgbg_obj)
        cv2.imshow('fgmask', resize(I_com,0.2)) 
        print(frame_id)
        k = cv2.waitKey(30) & 0xff
        #prv_regions = []
        if k == 27: 
            break
        ret, frame = cap.read()

    cap.release() 
    cv2.destroyAllWindows() 