
import numpy as np
import cv2

import argparse
from utils import read_tracks, save_tracks, resize, test_box, detect_overlaping

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

    # good tracks
    saved_tracks = []

    candidates_objs = []

    # run first frame logic
    previous_frame = frame.copy()
    bg = previous_frame.copy()
    #Fix_obj = FixView(bg)
    color_map =[(0,255,0), # ped
                (255,0,0), # cyclist
                (0, 0, 0)] # cars
    #ret, frame = v_obj.read()

    while frame is not None:

        if frame_id in tracking_data:
            objects_2_draw = tracking_data[frame_id]
        else:
            print('text finished before video')
            break

        for obj in objects_2_draw:
            box,class_id,track_id = obj
            #print(box)

            cv2.rectangle(frame,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),color=color_map[class_id-1],thickness=4)
            cv2.putText(frame,str(track_id),(box[0],box[1]),2,3,color=color_map[class_id-1],thickness=4)

        cv2.imshow('fgmask', resize(frame,0.3)) 
        k = cv2.waitKey(30) & 0xff
        #prv_regions = []
        if k == 27: 
            break


    # save the most good tracks
    #detections = detector.detect(previous_frame)
    #for obj in objects:
    
    #    Track,Save = obj.update()
    #
    #     if Track and (obj.class_id != -1):
    #        saved_tracks.append(obj)
        #ok,detections = obj.filter_by_detections(detections)
        #if ok or obj.good_enough():
        #    saved_tracks.append(obj)


        ret, frame = v_obj.read()
        frame_id += 1

    cv2.destroyAllWindows()



if __name__=='__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")
    #ap.add_argument("-t", "--tracker", type=str, default="kcf",
    #	help="OpenCV object tracker type")

    args = vars(ap.parse_args())
    main(args)


