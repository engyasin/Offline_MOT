# Python code for Background subtraction using OpenCV 
import numpy as np 
import cv2 
from skimage.measure import label, regionprops
from utils import resize

from config import config

class BG_substractor():
    """
    A class to perform background subtraction on videos
    based on opencv implementation.

    ...

    Attributes
    ----------
    history : int
        Number of frames to calculate the background from.
    thresh : int
        Threshold on the squared distance between the pixel 
        and the sample to decide whether a pixel is close to that 
        sample. This parameter does not affect the background 
        update.
    shadows : bool
        whether to detect the shadows or not.
    fgbg : class instance
        The background subtraction object of type KNN
    bg : numpy array
        The current calculated background
    kernel : numpy array
        The kernel array to do erosion process on the resulting foreground

    Methods
    -------
    bg_substract(numpy array) -> numpy array
        Process a new frame to find the foreground

    get_big_objects(numpy array,numpy array) -> (numpy array,list)
        Process a foreground with its frame to get the group
        of the different background objects.
    """
    def __init__(self,bg):
        """
        Parameters
        ----------
        bg : numpy array
            The background object for the first time

        """

        self.history = config.bgs_history
        self.thresh  = config.bgs_threshold
        self.shadows = config.bgs_shadows
        self.fgbg = cv2.createBackgroundSubtractorKNN(self.history,self.thresh,self.shadows)
        self.bg = bg.copy()

        kernel_size = config.bgs_erosion_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))

    def bg_substract(self,frame):
        """Find out the background and foreground and post process them

        It applies the background subtraction object several times on 
        the background to focus more on the last changes. Then it erode
        the resulting foreground.

        Parameters
        ----------
        frame : numpy array
            The image whose foreground should be found.

        Returns
        ------
        numpy array
            The foreground of the frame as grayscale image
        """
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
        if self.shadows:
            fgmask[fgmask<255] = 0

        self.bg = self.fgbg.getBackgroundImage()
        return fgmask

    def get_big_objects(self,fg_mask,frame):
        """Find the foreground objects based on the foreground image

        Parameters
        ----------
        fg_mask : numpy array
            The forground as grayscale image
        frame : numpy array
            The input image related to the foreground

        Returns
        -------
        (numpy array, list)
            A tuple of a new foreground image, and list of foreground 
            objects.

        """
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

    cap = cv2.VideoCapture('../../DJI_0148.mp4')
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