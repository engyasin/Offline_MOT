import argparse
import numpy as np
import cv2

from config import config
from fix_view import FixView
from tracking import track_objs, check_tracking#, track_new, TrackerObj
from background_substraction import BG_substractor

#from Detection.detection import detect
from detection import YoloDetector

from utils import save_tracks, resize, test_box, detect_overlaping

from objects_classes import  TrafficObj





def detections_to_objs(detections,frame,frame_id):
    # results : (p1,p2,prob,class_id)
    output = []
    for obj_item in detections:
        if obj_item[2]>config.detect_thresh:
            box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]
            if all(box): output.append(TrafficObj(frame,frame_id,box,-1,class_id=obj_item[3],detection_way=1))
    return output


def bgObjs_to_objs(bgObjs,frame,frame_id):
    # results : (p1,p2,prob,class_id)
    output = []
    for obj_item in bgObjs:
        box = [obj_item.bbox[1],obj_item.bbox[0],obj_item.bbox[3]-obj_item.bbox[1],obj_item.bbox[2]-obj_item.bbox[0]]

        if all(box): 
            new_obj = TrafficObj(frame,frame_id,box,-1,class_id=-1,detection_way=3)
            new_obj.set_theta(obj_item.orientation)
            output.append(new_obj)
    return output

def FirstFrame(frame):

    # detect
    detector  = YoloDetector('yolov4-obj.cfg','Yolov4_epoch300.pth',use_cuda=False)
    # 'Yolov4_epoch300.pth'
    #  'yolov4_last.pth'

    results,img_wh = detector.detect(frame)
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
            print(box)
            if test_box(box,img_wh): 
                print('taken')
                output.append(TrafficObj(frame,0,box,Track_id,class_id=obj_item[3],detection_way=1,detect_prob=obj_item[2]))
                Track_id += 1
        #print('+++++++ ',obj_item)

    return output, detector

def add_new_objs_from_bg(frame,frame_id,regions):

    objs = []
    for r in regions:
        box = [r.bbox[1],r.bbox[0],r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0]]
        #to remove objects at edge
        if all(box):
            objs.append(TrafficObj(frame,frame_id,box,class_id=-1))
    return objs

def link_obj_2_detect(obj, detections):

    obj_cntr = int(obj.box[0]+(obj.box[2]/2)),int(obj.box[1]+(obj.box[3]/2))
    detections_dists = []
    for obj_item in detections:

        if obj_item[3] != obj.class_id:
            #TODO what about unknown case
            detections_dists.append(1e9)
            continue

        dist = np.linalg.norm(((obj_item[0][0]+obj_item[1][0])//2 - obj_cntr[0],
                                 (obj_item[0][1]+obj_item[1][1])//2 - obj_cntr[1]))
        detections_dists.append(dist)

    Ok = min(detections_dists) < 80
    if Ok:
        obj_item = detections.pop(np.argmin(detections_dists))
        box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]
        obj.box = box
        obj.boxes.append(box)

    return Ok, obj, detections 

def find_obj_indx_set(obj,frame):
    w,h = frame.shape
    indx_arr = np.arange(w*h).reshape(w,h)
    indx_obj = indx_arr[obj.box[1]:obj.box[1]+obj.box[3],obj.box[0]:obj.box[0]+obj.box[2]]
    return set(map(lambda y: tuple(y),tuple(np.array(np.unravel_index(indx_obj.ravel(),(w,h))).T)))


def main(args):

    # read video file
    v_obj = cv2.VideoCapture(args["video"])
    ret, frame = v_obj.read()
    frame_id = 0

    # good tracks
    saved_tracks = []

    candidates_objs = []

    # run first frame logic
    objects,detector = FirstFrame(frame)
    print('len_first_obj ',len(objects))
    previous_frame = frame.copy()
    bg = previous_frame.copy()
    #Fix_obj = FixView(bg)
    BG_s = BG_substractor(bg)

    ret, frame = v_obj.read()
    #foreground = BG_s.bg_substract(frame)
    # for every frame and object in the list:

    while frame is not None:
        frame_id += 1

        # stabilize frame by frame
        #frame = Fix_obj.fix_view(frame,fgmask=foreground)

        # bg substract
        foreground = BG_s.bg_substract(frame)

        # track
        objects = track_objs(frame,frame_id,objects)
        candidates_objs = track_objs(frame,frame_id,candidates_objs)

        # add new from BG
        foreground , new_bg_objects = BG_s.get_big_objects(foreground,frame)

        # check tracking with background substraction

        #all_ok, objects, new_foreground = check_tracking(objects,foreground, new_objects)


        # track everything if new objects are added by bg_substract

        #objects.extend(add_new_objs_from_bg(frame,frame_id,new_objects))

        #if not all_ok:
        #    objects = track_new(objects)
        # update objects
        # TODO no need if a gloabl detection will be done
        done_detect = False
        lost_indx = []
        for i,obj in enumerate(objects+ candidates_objs):
            #obj.update()

            if not(obj.tracking_state[-1]):# or obj.need_redetect:
                # failed tracking
                if not(done_detect):
                    detections,_ = detector.detect(frame)
                    done_detect = True
                ok,detections = obj.filter_by_detections_dist(detections)
                obj.set_detection(ok)
                if ok:
                    obj.re_init_tracker(frame)
                    obj.tracking_state[-1] = True
                else: 
                    lost_indx.append(i)
                # if not ok, it could be out of screen

            ok_bg,new_bg_objects = obj.filter_by_bg_objs(new_bg_objects)
            obj.set_bg_substract(ok_bg)

            #else:
            #    ok,detections = obj.filter_by_detections_dist(detections)
            #    if not(ok): lost_indx.append(i)
            # deal with the newly not detected with spical logic
        print('len of lost_indx: ',len(lost_indx))
        # check with bg substract
        #for i,obj in enumerate(objects+candidates_objs):
        #    ok,new_bg_objects = obj.filter_by_bg_objs(new_bg_objects)
        #    obj.checked_with_bg(ok)

        # add new objects from BG stage:
        # find current forground:

        curr_fg = np.zeros_like(foreground)
        #TODO combine with first loop , with deleting instead of update

        # maybe make object thiner to allow for nearby object to be detected
        br = 5 # 5 pixels for borders
        for obj in objects+candidates_objs:

            curr_fg[obj.box[1]+br:obj.box[1]+obj.box[3]-br,obj.box[0]+br:obj.box[0]+obj.box[2]-br] = 1

        for n_obj in bgObjs_to_objs(new_bg_objects,frame,frame_id):
            # or sum bigger than thresh
            if curr_fg[n_obj.box[1]:n_obj.box[1]+n_obj.box[3],n_obj.box[0]:n_obj.box[0]+n_obj.box[2]].any():
                continue
            else:
                print('new obj added')
                candidates_objs.append(n_obj)
                #objects.append(n_obj)
        """
        for i in lost_indx:
            print('++++++ ',len(detections), len(lost_indx))
            if len(detections):
                ok, objects[i], detections = link_obj_2_detect(objects[i] ,detections)
                if ok:
                    objects[i].re_init_tracker(frame)
                    objects[i].tracking_state[-1] = True

        """

            # filter bad objects

        # detect every N frame,
        if (frame_id%config.detect_every_N)==0:
            if not(done_detect):
                detections, _ = detector.detect(frame)

            # filter bad objects after detection
            #new_objects = []
            print('len objs : ',len(objects))
            print('len detections : ',len(detections))
            for obj in objects:
                # object deleted if not ok here:
                ok,detections = obj.filter_by_detections_dist(detections,check=True)
                obj.set_detection(ok)
                print(len(detections),ok)
                if ok:
                    obj.re_init_tracker(frame)
                    #new_objects.append(obj)
                #elif obj.good_enough():
                    # not ok, but have long track
                #    saved_tracks.append(obj)
            if len(detections):
                print('Theres a problem. some objects detected but never seen before')
            # TODO replace with saving and delation
            #objects = new_objects[:]

            # TODO add logic for saving long trajectories before deleting
            # add new detections (if prob high enough)

            ## check if new detection agree with bg:
            new_candidates_objs = []
            print('Creating new objects **** ')
            for bg_obj in candidates_objs:
                if len(detections)==0: break
                ok,detections = bg_obj.filter_by_detections_dist(detections,check=True)
                print(ok)
                bg_obj.set_detection(ok)
                if ok:
                    bg_obj.re_init_tracker(frame)
                    bg_obj.set_track_id(max([x.track_id for x in objects+saved_tracks]+[0])+1)
                    objects.append(bg_obj)
                else:
                    new_candidates_objs.append(bg_obj)
            candidates_objs = new_candidates_objs[:]
            #for detect in detections:
            #    pass
            #objects.extend(detections_to_objs(detections,frame,frame_id))
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
                continue

        objects, candidates_objs = new_objs[:], new_candidates_objs[:]
        while True:
            to_remove = detect_overlaping(objects,overlap_thresh=config.overlap_thresh)
            if to_remove == -1:
                break
            _ = objects.pop(to_remove)

        previous_frame = frame.copy()

        for obj in objects+candidates_objs:
            frame = obj.draw(frame)
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
