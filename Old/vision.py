import numpy as np
import cv2

def scale_img(img,scale=0):
    return cv2.resize(img,dsize=(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_AREA)



def downscale_video(filename=''):
    cap_read = cv2.VideoCapture(filename)
    ret,frame = cap_read.read()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap_write = cv2.VideoWriter('new_copy.avi',fourcc, 30,(frame.shape[1], frame.shape[0]), True)

    while ret:

        cap_write.write(scale_img(frame,0.25))
        ret,frame = cap_read.read()
    cap_write.release()
    cap_read.release()