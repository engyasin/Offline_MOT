import cv2
import numpy as np
import logging, os
from background_subtraction import BG_subtractor 

from config import configs
from utils_ import resize

class FixView():

    """
    A class to fix the frames postions in a video
    that experiance shaking or random movement.

    ...

    Attributes
    ----------
    sift : SIFT detector object
        The sift detector object that will perform the detection 
        on frames
    flann : FLANN based matcher object
        The FLANN matcher with parameters compitaple with matching
        SIFT keypoints
    mask_bg : numpy array
        The boolean mask which determines the acceptable area
        to detect keypoints from in the input image.
    kps_bg : list of KeyPoint objects
        The background keypoints that will be matched with 
        each next frame.
    des_bg : numpy array
        The background descriptors of the kypoints to perform 
        matching later
    min_matches : int
        Minumum Number of matches to perform transformation on fame
    kernel : numpy array
        The kernel array to do dilation process on the foreground
        to form the mask

    Methods
    -------
    get_good_matches(list) -> list
        Filter the matches objects according to the distance and distance ratio test.

    fix_view(numpy array,numpy array) -> numpy array
        Process the frame by detecting the 2D affine motion, and transform it
        if it is found with enough number of matches points 
    
    set_new_bg(numpy array, numpy array) -> None
        Set a new background image, keypoints and descriptors according to the
        forground if provided

    """

    def __init__(self,bg_rgb, config= configs()):
        """
        Parameters
        ----------
        bg_rgb : numpy array
            The background image that all the next frames will be refrenced to,
            whether they need transformation or not.
        config : config instance 
            A class instance of all the configuration parameters

        """
        # Initiate SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=2000)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 500)

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)

        bg = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2GRAY)

        # taking from the border to avoid, foreground
        mask =  np.zeros_like(bg,dtype=np.uint8)

        borders_ = int(np.min(mask.shape)/2)
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
        """Filter the matches objects according to the distance and distance ratio test

        Parameters
        ----------
        matches : list
            The list of Matches objects list refering to the same descriptor and
            all its matches

        Returns
        -------
        list
            a list of the matches objects passing the distance ratio and sorted
            according to the distance of length no more than 500

        """
        good = []
        for mn in matches:
            if len(mn)==2:
                m,n=mn
                if m.distance < 0.7*n.distance:
                    good.append(m)
            else:
                #try adding anyway, to be later tested with distance
                good.append(mn[0])
        # take only the best 500
        good = sorted(good,key=lambda x: x.distance)[:500]
        return good

    def fix_view(self,frame,fgmask=None):
        """Process the frame by detecting the 2D affine motion,
        and transform it if it is found with enough number of matches 
        points 

        Parameters
        ----------
        frame : numpy array
            The frame that should be transformed to the fixed postion
        fgmask : numpy array, optional
            The mask defining where to detect the frame keypoints 
            and descriptors to match with

        Returns
        -------
        numpy array
            The transformed frame if the conditions apply.

        """

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
                logging.warning("Not transformed, there is low number of correcr matches")
        else:
            logging.warning( "Not enough matches are found - {}/{}".format(len(good), self.min_matches) )

        return frame

    def set_new_bg(self,bg_rgb,fg_mask=None):
        """Set a new background image, keypoints and descriptors
        according to the forground if provided

        Parameters
        ----------
        bg_rgb : numpy array
            The new background to match each frame with
        fg_mask : numpy array, optional
            The mask defining where to detect the background keypoints 
            and descriptors

        """
        bg = cv2.cvtColor(bg_rgb, cv2.COLOR_BGR2GRAY)
        if fg_mask is None:
            mask = self.mask_bg.copy()
        else:
            fg_img = cv2.dilate(fg_mask,self.kernel,iterations = 1)
            mask = np.zeros_like(fg_img,dtype=np.uint8)
            mask[fg_img==0] = 255
        self.kps_bg,self.des_bg = self.sift.detectAndCompute(bg,mask)


if __name__ == '__main__':

    cap = cv2.VideoCapture(os.path.join(configs.cwd,'model','sample.mp4'))
    frame_id = 1
    cap.set(1, frame_id-1)
    ret,bg_rgb = cap.read()

    Fix_obj = FixView(bg_rgb)
    BG_s = BG_subtractor(bg_rgb)

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