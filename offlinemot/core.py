"""Perform the main loop of reading the video and tracking the traffic
objects in it.

This script reads the video file frame by frame and tries to detect 
all the objects in it and track them. Additionally, a check with
background subtraction is done.

This contains the following functions:

    * main - the main function of the script
    * track_objs - Perfrom tracking on every object in the frame
    * bgObjs_to_objs - transform background subtraction objects to traffic objects
    * FirstFrame - initilize the detector and return the objects in the first frame if they fall within the image borders
    * detections_to_objects - Transfrom a group of detection results in one frame to traffic objects instances
    * set_params - Open the configuration file in a text editor to set parameters
"""

import argparse ,logging ,os
import webbrowser
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)

from config import configs
from fix_view import FixView
from background_subtraction import BG_subtractor

from detection import YoloDetector
from utils_ import save_tracks, resize, check_box, detect_overlaping, transform_detection

from objects_classes import  TrafficObj
from post_process import postProcessAll

def createTracker():

    if configs.Tracker_goturn:
        def tracker_():
            params = cv2.TrackerGOTURN_Params()

            params.modelTxt = os.path.join(configs.cwd,'model','goturn.prototxt')
            params.modelBin = os.path.join(configs.cwd,'model','goturn.caffemodel')

            tracker = cv2.TrackerGOTURN_create(params)

            return tracker
    else:
        tracker_ = cv2.TrackerKCF_create
    
    return tracker_


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


def bgObjs_to_objs(bgObjs,frame,frame_id,config=configs()):
    """Transform background subtraction objects to traffic objects

    Parameters
    ----------
    bgObjs : list
        a list of the new moving objects in the current frame.
        excluding the objects where the objects already detected.
    frame : numpy array
        The input image where the objects are detected
    frame_id : int
        The order of the frame in the video
    config : config instance 
        A class instance of all the configuration parameters

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
            new_obj = TrafficObj(frame,frame_id,box,-1,config,tracker=createTracker(),class_id=-1,detection_way=3)
            output.append(new_obj)
    return output


def detections_to_objects(detections,frame,config,last_track_id=0):
    """Convert a list of detections out of Yolo network to traffic
    object class instances.

    Parameters
    ----------
    detections : list
        A list of detections
    frame : numpy array
        The input image in rgb format which the detections are in
    config : config instance 
        A class instance of all the configuration parameters
    last_track_id: int
        The id number from which numbering is starting from
        (default 0)

    Returns
    -------
    list
        a list of the traffic object classes representing the
        new detections in the input

    """
    output = []
    Track_id = last_track_id 
    img_wh = frame.shape[:-1]
    for obj_item in detections:
        if obj_item[2]>config.detect_thresh:
            box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]

            if check_box(box,img_wh): 

                output.append(TrafficObj(frame,0,box,Track_id,config,tracker=createTracker(),class_id=obj_item[3],detection_way=1,detect_prob=obj_item[2]))
                Track_id += 1

    return output 

def FirstFrame(frame, config):
    """initilize the detector and return the objects in the first frame
    if they fall within the image borders

    Parameters
    ----------
    frame : numpy array
        The first frame of the video
    config : config instance 
        A class instance of all the configuration parameters

    Returns
    -------
    tuple
        a tuple of a list of the new detected traffic objects as the first
        element and the detector instance of YOLOv4 detector as the second
        element.

    """
    # detect
    detector  = YoloDetector(config)

    results, img_wh = detector.better_detection(frame)
    # create objects based on detections
    # results : (p1,p2,prob,class_id)
    output = detections_to_objects(results,frame,config)

    if config.manual_start:
        frame_ = frame.copy()
        for obj in output:
            if obj.last_detect_prob > configs.detect_thresh:
                print(f'Id {obj.track_id} probabilty: {np.round(obj.last_detect_prob,3)}')
                frame_ = obj.draw(frame_)
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 1100, 600)
        manual_boxes = cv2.selectROIs('frame',frame_,fromCenter=False,showCrosshair=False)

        cv2.destroyWindow('frame')

        for box in manual_boxes:
            results.insert(0,[(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),10.0,1])

        #cv2.imshow('fgmask', resize(frame,config.resize_scale)) 
        output = detections_to_objects(results,frame,config)


    return output, detector

def enhance(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def set_params():
    """Open the configuration file in a text editor
    to set parameters
    """
    webbrowser.open(os.path.join(configs.cwd,'config.py'))

def extract_paths(vid_name=None, config=configs()):
    """The main loop to detect and track traffic objects in the video
    and save the result after post processing to a text file.

    Parameters
    ----------
    vid_name : str
        The path to the video name which should be processed
    config : config instance 
        A class instance of all the configuration parameters

    """

    ###### Step 1: Initilaize parmeters, video reader,
    ###### and classes instances

    # read video file
    if vid_name is None:
        vid_name = os.path.join(configs.cwd,'model','sample.mp4')

    v_obj = cv2.VideoCapture(vid_name)

    #v_obj.set(cv2.CAP_PROP_POS_FRAMES,180)
    _ , frame = v_obj.read()
    #frame = enhance(frame)
    frame_id = 0

    # good tracks
    saved_tracks = []
    candidates_objs = []
    deleted_tracks = []

    config.print_summary()

    # run first frame logic
    objects,detector = FirstFrame(enhance(frame),config)

    logging.info(f'Number of detected objects in first frame: {len(objects)}')

    previous_frame = frame.copy()
    bg = previous_frame.copy()

    Fix_obj = FixView(bg,config)
    BG_s = BG_subtractor(bg,config)

    ret, frame = v_obj.read()
    #frame = enhance(frame)

    foreground = BG_s.bg_substract(frame)
    # for every frame and object in the list:

    while ret:

        frame_id += 1
        #frame = frame[:,:2800,:]

        logging.info(f'Frame: {frame_id},Time: {np.round(frame_id/30,2)}')

        ###### Step 2: Stabilize frame by frame
        if config.do_fix: 
            frame = Fix_obj.fix_view(frame,fgmask=foreground)

        ######## Step 3: Backgorund Substract
        foreground = BG_s.bg_substract(frame)

        ######## Step 4: Track using OpenCV trackers
        objects = track_objs(frame,frame_id,objects)
        candidates_objs = track_objs(frame,frame_id,candidates_objs)

        ####### Step 5: Filter the small object in the background 
        ####### subtraction result
        foreground , new_bg_objects = BG_s.get_big_objects(foreground,frame)


        done_detect = False
        lost_indx = []
        curr_fg = np.zeros_like(foreground)

        # maybe make object thiner to allow for nearby object to be detected
        br = config.bgs_broder_margin # pixels for borders

        ########## Step 6: find failing tracking cases and confirm it with a
        ########## a detection, confirm eveything with background subtraction
        ########## and build binary mask for confirmed objects.

        for i,obj in enumerate(objects+ candidates_objs):

            if not(obj.tracking_state[-1]) and (frame_id%config.detect_every_N):
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
                        obj.set_track_id(max([x.track_id for x in objects+saved_tracks+deleted_tracks]+[-1])+1)
                        objects.append(obj)
                        logging.info('Creating new objects...  ')
                else: 
                    lost_indx.append(i)

                # if not ok, it could be out of screen
            # check tracking with background subtraction

            ok_bg,new_bg_objects = obj.filter_by_bg_objs(new_bg_objects)
            obj.set_bg_substract(ok_bg)
            if ok_bg and not(obj.tracking_state[-1]):
                obj.re_init_tracker(frame)
                obj.tracking_state[-1] = True

            br_w,br_h = int(obj.box[2]*br),int(obj.box[3]*br)
            obj.box = tuple([int(b) for b in obj.box])
            curr_fg[obj.box[1]+br_h:obj.box[1]+obj.box[3]-br_h,obj.box[0]+br_w:obj.box[0]+obj.box[2]-br_w] = 1
            # deal with the newly not detected with spical logic

        ######## Step 7: Add new objects from BG subtraction
        ########  stage to candiatdates list

        for n_obj in bgObjs_to_objs(new_bg_objects,frame,frame_id,config):
            # or sum bigger than thresh
            if curr_fg[n_obj.box[1]:n_obj.box[1]+n_obj.box[3],n_obj.box[0]:n_obj.box[0]+n_obj.box[2]].any():
                # box overlap
                continue
            else:
                logging.info('New object is added')
                candidates_objs.append(n_obj)
                #objects.append(n_obj)

        ####### Step 8: detect every N frame, check all (objects and candidates)
        ####### move from candidates to objects if checked with detections.

        if (frame_id%config.detect_every_N)==0:
                
            detections, _ = detector.better_detection(enhance(frame),additional_objs=candidates_objs+objects)
            for obj in objects:
                # object deleted if not ok here:
                ok,detections = obj.filter_by_detections_dist(detections,check=True)
                obj.set_detection(ok)
                if ok:
                    obj.re_init_tracker(frame)

            ## check if new detection agree with bg:

            for bg_obj in candidates_objs:

                # not taken objects? ==> detect on higher scale
                # frame shape is h,w,3
                # maybe when first an object is created

                if len(detections)==0: 
                    break
                ok,detections = bg_obj.filter_by_detections_dist(detections,check=True)
                bg_obj.set_detection(ok)
                if ok:
                    bg_obj.re_init_tracker(frame)
                    bg_obj.set_track_id(max([x.track_id for x in objects+saved_tracks+deleted_tracks]+[-1])+1)
                    logging.info('Creating new objects... ')
                    objects.append(bg_obj)

            # remove the detected classes
            candidates_objs = [candi_obj for candi_obj in candidates_objs if(candi_obj.class_id ==-1)]

            if len(detections):
                logging.warning('Some objects are detected but not moving or seen before')

        ######### Step 9: check all objects if good enough to continue 
        ######### tracking, to save if long and confirmed but lost, 
        ######### or to delete.
        new_objs , new_candidates_objs = [], []
        for obj in objects+candidates_objs:
            Track,Save = obj.update()
            if Track:
                if obj.class_id == -1:
                    new_candidates_objs.append(obj)
                else:
                    new_objs.append(obj)
            elif Save and obj.class_id != -1:
                saved_tracks.append(obj)
            else:
                logging.info('Object {} is deleted because of not enough detections'.format(obj.track_id))
                deleted_tracks.append(obj)
                # delete
                continue

        objects, candidates_objs = new_objs[:], new_candidates_objs[:]

        ######## Step 10: check if any objects are overlapping with another
        ######## Delete if a minmum area is found
        all_objects = objects+candidates_objs
        while True:
            to_remove = detect_overlaping(all_objects,overlap_thresh=config.overlap_thresh,overlap_steps=config.overlap_steps)
            if to_remove == -1:
                break

            if (np.array(all_objects[to_remove].trust_level).sum()>(config.min_history+5)) and \
                all_objects[to_remove].tracking_state[-1]:
                # true object?
                # return it to its last ok state.
                all_objects[to_remove].box = all_objects[to_remove].boxes[-2]
                all_objects[to_remove].boxes[-1] = all_objects[to_remove].boxes[-2]
                all_objects[to_remove].tracking_state[-1] = False
            
                continue

            logging.info('Object {} is deleted because of overlaping'.format(all_objects[to_remove].track_id))
            _ = all_objects.pop(to_remove)
            if to_remove<len(objects):
                deleted_tracks.append(objects.pop(to_remove))
            else:
                deleted_tracks.append(candidates_objs.pop(to_remove-len(objects)))
                
        ######## Step 11: Draw each frame with the tracking result
        ######## if needed for debugging

        if config.draw:
            for obj in objects+candidates_objs:
                frame = obj.draw(frame)
            cv2.namedWindow('fgmask', cv2.WINDOW_NORMAL)
            cv2.imshow('fgmask', resize(frame,config.resize_scale)) 
            k = cv2.waitKey(10) & 0xff
            if k == 27: 
                break
            elif k in [ord('s'),ord('S')] :# or (frame_id%45)==0:
                print('manaul correction')
                #candidate objects maybe useless
                candidates_objs = []
                manual_boxes = cv2.selectROIs('fgmask',frame,fromCenter=False,showCrosshair=False)
                #cv2.destroyWindow('fgmask')
                results = []
                for box in manual_boxes:
                    results.insert(0,[(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),2.0,1])
                for obj in objects:
                    ok,results = obj.filter_by_detections_dist(results,check=True)
                    obj.set_detection(ok)
                    if ok:
                        obj.re_init_tracker(frame)
                        obj.tracking_state[-1] = True
        # NOTE 
        #for _ in range(15):
        ret, frame = v_obj.read()
        #frame = enhance(frame)
        #print(f' frame Nr: {frame_id}')

    ###### Step 12: Take the current and saved objects,
    ###### smooth the trajectories, interpolate the missing points,
    ###### find the angels and save to text file.
    for obj in objects:
        Track,Save = obj.update()
        if Track and (obj.class_id != -1):
            saved_tracks.append(obj)
    for obj in deleted_tracks:
        Track,Save = obj.update()
        if (Save or Track) and (obj.class_id != -1):
            saved_tracks.append(obj)

    cv2.destroyAllWindows()
    v_obj.release()
    del Fix_obj
    del BG_s

    saved_tracks = postProcessAll(saved_tracks,config)
    save_tracks(saved_tracks,vid_name)

if __name__=='__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("-v", "--video", type=str, default= "model/sample.mp4",
        help="path to input video file")

    args = vars(ap.parse_args())
    extract_paths(args['video'])
