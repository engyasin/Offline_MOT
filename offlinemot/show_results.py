"""Show the result of the tracking of a video

This script shows the results of the tracking and load it form a saved 
text file. The video file name and text file name should be the same.
The video directory should be passed after -v flag in the command line.

This script requires that `opencv-python` be installed within the Python
environment you are running this script in.

This contains the following functions:

    * main - the main function of the script
"""

import numpy as np
import cv2

import argparse
from offlinemot.config import config
from utils import read_tracks, resize

# read video from args

# read text and save it in a dict

# load each video frame and draw on it.




def main(args):
    # read video file
    v_obj = cv2.VideoCapture(args["video"])

    tracking_data = read_tracks(args["video"])
    # frame_id: ([x,y,w,h],class)

    ret, frame = v_obj.read()
    frame_id = 0

    # run first frame logic
    #Fix_obj = FixView(bg)
    color_map =[(0,255,0), # ped
                (255,0,0), # cyclist
                (0, 0, 0)] # cars
    #ret, frame = v_obj.read()
    while ret:#frame is not None:

        if frame_id in tracking_data:
            objects_2_draw = tracking_data[frame_id]
        else:
            objects_2_draw = []
            #print('text finished before video')
            #break

        for obj in objects_2_draw:
            box,class_id,track_id,angel = obj
            #print(box)

            #TODO draw rotated
            center = int(box[0]+(box[2]/2)),int(box[1]+(box[3]/2))
            rect = cv2.boxPoints((center,(box[2],box[3]),angel))
            rect = np.intp(rect)
            cv2.drawContours(frame,[rect],0,color=color_map[class_id-1],thickness=4)
            #cv2.drawContours(stabilized_frame, [rect], 0, (255,0,0),4)   

            #cv2.rectangle(frame,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color=color_map[class_id-1],thickness=4)
            cv2.putText(frame,str(track_id),(box[0],box[1]),2,3,color=color_map[class_id-1],thickness=4)
        cv2.imshow('fgmask', resize(frame,config.resize_scale)) 
        k = cv2.waitKey(30) & 0xff
        #prv_regions = []
        if k == 27: 
            break

        ret, frame = v_obj.read()
        frame_id += 1

    cv2.destroyAllWindows()
    v_obj.release()



if __name__=='__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    #ap.add_argument("-t", "--tracker", type=str, default="kcf",
    #	help="OpenCV object tracker type")

    args = vars(ap.parse_args())
    main(args)


