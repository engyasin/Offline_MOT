import cv2
import numpy as np
import imutils

def simplifyimg(img,parts=6):

    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

    ret,label,center=cv2.kmeans(Z,parts,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image

    center = np.uint8(center)
    #
    res = center[label.flatten()]
    return center,res.reshape((img.shape))

    """
    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """



def main():
    vidcap = cv2.VideoCapture('../../DJI_0148.mp4')

    grabbed_frame, frame = vidcap.read()

    centers,seg = simplifyimg(frame,parts=25)
    seg = imutils.resize(seg, width=600)
    breakpoint()
    cv2.imshow('res2',seg)
    cv2.waitKey(0)
    vidcap.release()
    cv2.destroyAllWindows()

