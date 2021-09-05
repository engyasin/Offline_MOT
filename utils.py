import cv2
import numpy as np


def resize(img,scale=1):
    if len(img.shape)==2:
        img = np.dstack((img,img,img))

    I_com = cv2.resize(img,
                        tuple([int(x*scale) for x in img.shape[::-1][1:]]))
    return I_com