"""Perform the main loop of reading the video and tracking the traffic
objects in it.

This script reads the video file frame by frame and tries to detect 
all the objects in it and track them. Additionally, a check with
background substraction is done.

This contains the following functions:

    * main - the main function of the script
    * track_objs - Perfrom tracking on every object in the frame
    * bgObjs_to_objs - transform background substraction objects to traffic
    objects
    * FirstFrame - initilize the detector and return the objects in the
    first frame if they fall within the image borders

"""

import argparse
import numpy as np
import cv2

from config import config
from fix_view import FixView
from background_substraction import BG_substractor

#from Detection.detection import detect
from detection import YoloDetector
from utils import find_overlap
from utils import save_tracks, resize, check_box, detect_overlaping, transform_detection

from objects_classes import  TrafficObj
from post_process import postProcessAll


def track_objs(frame,frame_id,objects):
    """Perfrom tracking on every object in the frame

    Parameters
    ----------
    frame : numpy array
        The input image to track the objects within
    frame_id : int
        The order of the frame in the video
    objects : list
        list of the current traffic object under tracking
        to check where they moved in the frame.

    Returns
    -------
    list
        a list of the traffic object classes with updated
        postions according to the tracking result.

    """
    # return new object
    for obj in objects:
        obj.track(frame,frame_id)

    return objects


def bgObjs_to_objs(bgObjs,frame,frame_id):
    """Transform background substraction objects to traffic objects

    Parameters
    ----------
    bgObjs : list
        a list of the new moving objects in the current frame.
        excluding the objects where the objects already detected.
    frame : numpy array
        The input image where the objects are detected
    frame_id : int
        The order of the frame in the video

    Returns
    -------
    list
        a list of traffic object instances, representing the 
        previous list of objects in the input if within the image

    """
    # results : (p1,p2,prob,class_id)
    output = []
    for obj_item in bgObjs:
        box = [obj_item.bbox[1],obj_item.bbox[0],obj_item.bbox[3]-obj_item.bbox[1],obj_item.bbox[2]-obj_item.bbox[0]]

        if all(box): 
            new_obj = TrafficObj(frame,frame_id,box,-1,class_id=-1,detection_way=3)
            output.append(new_obj)
    return output

def FirstFrame(frame):
    """initilize the detector and return the objects in the first frame
    if they fall within the image borders

    Parameters
    ----------
    frame : numpy array
        The first frame of the video

    Returns
    -------
    tuple
        a tuple of a list of the new detected traffic objects as the first
        element and the detector instance of YOLOv4 detector as the second
        element.

    """
    # detect
    detector  = YoloDetector(config.model_config,config.model_name,use_cuda=config.use_cuda)
    # 'Yolov4_epoch300.pth'
    #  'yolov4_last.pth'

    results,img_wh = detector.better_detection(frame)
    # create objects based on detections
    # results : (p1,p2,prob,class_id)
    output = []
    Track_id = 0
    for obj_item in results:
        if obj_item[2]>config.detect_thresh:
            box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]

            ##### NOTE added
            #np_box = np.array(box)
            #box = abs(np_box*(np_box>0)).tolist()
            ######
            #print(box)
            if check_box(box,img_wh): 
                # if within the image
                #print('taken')
                output.append(TrafficObj(frame,0,box,Track_id,class_id=obj_item[3],detection_way=1,detect_prob=obj_item[2]))
                Track_id += 1
        #print('+++++++ ',obj_item)

    return output, detector

def main(args):
    """The main loop to detect and track traffic objects in the video
    and save the result after post processing to a text file.

    Parameters
    ----------
    args : dict
        The input parameters as dictionary with values for keys such as
        video name.

    """
    # read video file
    v_obj = cv2.VideoCapture(args["video"])

    _ , frame = v_obj.read()
    frame_id = 0

    # good tracks
    saved_tracks = []
    candidates_objs = []

    # run first frame logic
    objects,detector = FirstFrame(frame)

    print('Number of detected objects in first frame: ',len(objects))

    previous_frame = frame.copy()
    bg = previous_frame.copy()

    Fix_obj = FixView(bg)
    BG_s = BG_substractor(bg)

    ret, frame = v_obj.read()

    foreground = BG_s.bg_substract(frame)
    # for every frame and object in the list:

    while ret:
        frame_id += 1
        #frame = frame[:,:2800,:]
        print('Frame: %s'%frame_id,'Time: %s ,'%(frame_id/30))
        ## stabilize frame by frame
        if config.do_fix: 
            frame, = Fix_obj.fix_view(frame,fgmask=foreground)

        ## bg substract
        foreground = BG_s.bg_substract(frame)

        ## track
        ########## Oridinary track
        objects = track_objs(frame,frame_id,objects)
        candidates_objs = track_objs(frame,frame_id,candidates_objs)

        ## add new from BG
        foreground , new_bg_objects = BG_s.get_big_objects(foreground,frame)


        ## update objects
        # TODO no need if a gloabl detection will be done
        done_detect = False
        lost_indx = []
        curr_fg = np.zeros_like(foreground)

        # maybe make object thiner to allow for nearby object to be detected
        br = config.bgs_broder_margin # pixels for borders

        ########## if track failed,  check with detection, 
        ########## check all with bgs 
        for i,obj in enumerate(objects+ candidates_objs):
            #obj.update()

            if not(obj.tracking_state[-1]):# and obj.track_id!=-1:
                # failed tracking
                if not(done_detect):
                    detections,_ = detector.better_detection(frame,additional_objs=candidates_objs+objects)
                    done_detect = True
                ok,detections = obj.filter_by_detections_dist(detections)
                obj.set_detection(ok)
                if ok:
                    obj.re_init_tracker(frame)
                    obj.tracking_state[-1] = True
                    if obj.track_id == -1:
                        # candidate
                        obj.set_track_id(max([x.track_id for x in objects+saved_tracks]+[0])+1)
                        objects.append(obj)
                        print('Creating new objects **** ')
                else: 
                    lost_indx.append(i)
                # if not ok, it could be out of screen

            # check tracking with background substraction
            ok_bg,new_bg_objects = obj.filter_by_bg_objs(new_bg_objects)
            obj.set_bg_substract(ok_bg)

            br_w,br_h = int(obj.box[2]*br),int(obj.box[3]*br)
            curr_fg[obj.box[1]+br_h:obj.box[1]+obj.box[3]-br_h,obj.box[0]+br_w:obj.box[0]+obj.box[2]-br_w] = 1
            # deal with the newly not detected with spical logic
        #print('len of lost_indx: ',len(lost_indx))

        ######## add new objects from BG stage to candiatdate:

        for n_obj in bgObjs_to_objs(new_bg_objects,frame,frame_id):
            # or sum bigger than thresh
            if curr_fg[n_obj.box[1]:n_obj.box[1]+n_obj.box[3],n_obj.box[0]:n_obj.box[0]+n_obj.box[2]].any():
                # box overlap
                continue
            else:
                print('new object added')
                candidates_objs.append(n_obj)
                #objects.append(n_obj)

        # detect every N frame, check all
        # add from candidate to objects if checked with detection
        if (frame_id%config.detect_every_N)==0:
            #if not(done_detect):
                
            detections, _ = detector.better_detection(frame,additional_objs=candidates_objs+objects)

            # filter bad objects after detection
            #new_objects = []
            for obj in objects:
                # object deleted if not ok here:
                ok,detections = obj.filter_by_detections_dist(detections,check=True)
                obj.set_detection(ok)
                if ok:
                    obj.re_init_tracker(frame)
                    #new_objects.append(obj)
                #elif obj.good_enough():
                    # not ok, but have long track
                #    saved_tracks.append(obj)
            #objects = new_objects[:]

            ## check if new detection agree with bg:

            for bg_obj in candidates_objs:

                # not taken objects? ==> detect on higher scale
                # frame shape is h,w,3
                # maybe when first an object is created
                #x,y,w,h = tuple(bg_obj.box)
                #cropped_img = frame[max(y-margin_crop,0):y+h+margin_crop,max(x-margin_crop,0):x+w+margin_crop]
                #new_detections, _ = detector.detect(cropped_img)
                #p0 = (max(x-margin_crop,0),max(y-margin_crop,0))
                #detections.extend(transform_detection(p0,new_detections))


                if len(detections)==0: 
                    break
                ok,detections = bg_obj.filter_by_detections_dist(detections,check=True)
                bg_obj.set_detection(ok)
                if ok:
                    bg_obj.re_init_tracker(frame)
                    bg_obj.set_track_id(max([x.track_id for x in objects+saved_tracks]+[0])+1)
                    print('Creating new objects **** ')
                    objects.append(bg_obj)

            # remove the detected classes
            candidates_objs = [candi_obj for candi_obj in candidates_objs if(candi_obj.class_id ==-1)]

            if len(detections):
                print('Note: some objects detected but not moving or seen before')

        ######### check if all object good enough to track again, or else to save if long and sure enough
        new_objs , new_candidates_objs = [], []
        for obj in objects+candidates_objs:
            Track,Save = obj.update()
            if Track:
                if obj.class_id == -1:
                    new_candidates_objs.append(obj)
                else:
                    new_objs.append(obj)
            elif Save:
                saved_tracks.append(obj)
            else:
                # delete
                continue

        objects, candidates_objs = new_objs[:], new_candidates_objs[:]
        ######## check if any objects are overlapping
        while True:
            to_remove = detect_overlaping(objects,overlap_thresh=config.overlap_thresh)
            if to_remove == -1:
                break
            _ = objects.pop(to_remove)

        if config.draw:
            for obj in objects+candidates_objs:
                frame = obj.draw(frame)
            #for b in new_boxes:
            #   frame[b[1]:b[1]+b[3],b[0]:b[0]+b[2]] = 0 
            cv2.imshow('fgmask', resize(frame,0.2)) 
            k = cv2.waitKey(10) & 0xff
            #prv_regions = []
            if k == 27: 
                break

        ret, frame = v_obj.read()

    # save the most good tracks
    #detections = detector.detect(previous_frame)
    for obj in objects:
        Track,Save = obj.update()
        if Track and (obj.class_id != -1):
            saved_tracks.append(obj)
        #ok,detections = obj.filter_by_detections(detections)
        #if ok or obj.good_enough():
        #    saved_tracks.append(obj)

    cv2.destroyAllWindows()
    v_obj.release()
    del Fix_obj
    del BG_s

    saved_tracks = postProcessAll(saved_tracks)
    save_tracks(saved_tracks,args['video'])


if __name__=='__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", type=str,
        help="path to input video file")

    #ap.add_argument("-t", "--tracker", type=str, default="kcf",
    #	help="OpenCV object tracker type")

    args = vars(ap.parse_args())
    main(args)
