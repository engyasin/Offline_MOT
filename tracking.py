# import the necessary packages
import time
import cv2
import numpy as np
from utils import resize
from background_substraction import BG_substractor 
from fix_view import FixView

from config import config
from objects_classes import TrafficObj

def track_objs(frame,frame_id,objects):
    # return new object
    for obj in objects:
        obj.track(frame,frame_id)

    return objects

#         all_ok, objects = check_tracking(objects,foreground)

def check_tracking(objects,foreground, new_objects):
    #print(foreground.shape)
    #box is x,y,w,h
    all_ok = []
    w,h = foreground.shape
    indx_arr = np.arange(w*h).reshape(w,h)
    new_objs_sets = [set(map(lambda y: tuple(y),tuple(rgn.coords))) for rgn in new_objects]

    for obj in objects:
        r1,r2 = obj.box[1], obj.box[1] + obj.box[3]
        c1,c2 = obj.box[0], obj.box[0] + obj.box[2]
        #TODO delete all the wheit obj
        section = set(map(lambda y: tuple(y),tuple(np.array(np.unravel_index(indx_arr[r1:r2,c1:c2].ravel(),(w,h))).T)))
        obj_2_delete, max_intersection = 0,0
        for i,obj_coords in enumerate(new_objs_sets):
            if len(section.intersection(obj_coords)) > max_intersection:
                max_intersection = len(section.intersection(obj_coords))
                obj_2_delete = new_objects[i].slice
        # calculate distance from each
        if (max_intersection/(obj.box[2]*obj.box[3]))>config.bg_detect_thresh:
            foreground[obj_2_delete] = 0
            all_ok.append(True)
            obj.checked_with_bg(True)
        #breakpoint()
        else:
            all_ok.append(False)
            obj.checked_with_bg(False)
    
    return all_ok,objects, foreground


class TrackerObj:
    def __init__(self,frame,regions,type="kcf"):

        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.legacy.TrackerBoosting_create,
            "mil": cv2.legacy.TrackerMIL_create,
            "tld": cv2.legacy.TrackerTLD_create,
            "medianflow": cv2.legacy.TrackerMedianFlow_create,
            "mosse": cv2.legacy.TrackerMOSSE_create
            }

        #self.list_of_trackers = []
        #self.boxes = []
        #self.tracking_state = []
        self.objs = []

        for r in regions:
            box = [r.bbox[1],r.bbox[0],r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0]]

            #to remove objects at edge
            if all(box):
                #TODO add the type of the class (-1 means unknown)
                self.objs.append(TrafficObj(frame,0,box,tracker=self.OPENCV_OBJECT_TRACKERS[type],class_id=-1))
                #self.boxes.append(box)
                #self.tracking_state.append([])
                #self.list_of_trackers.append(self.OPENCV_OBJECT_TRACKERS[type]())
                # cv2.TrackerKCF_create())
                #self.list_of_trackers[-1].init(frame,box)

    
    def track(self,new_frame,frame_id):

        for i,obj in enumerate(self.objs):
            obj.track(new_frame,frame_id)

        #for i,tracker in enumerate(self.list_of_trackers):

        #    state, box = tracker.update(new_frame)
        #    #print(state)
        #    if state:
        #        #update boxes list
        #        self.boxes[i] = box

        #    self.tracking_state[i].append(state)

    def filter_obj(self):
        pass


    def draw(self,new_frame):

        # object that is being tracked
        for i,obj in enumerate(self.objs):
            new_frame = obj.draw(new_frame)
        #for i,box in enumerate(self.boxes):
            
        #    color_code = [(0, 0, 255),(0, 255, 0)][self.tracking_state[i][-1]]

        #    (x, y, w, h) = [int(v) for v in box]
        #    cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=color_code, thickness=4)

        return new_frame


if __name__ == '__main__':

    # TODO add the tracker object to each object class

    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations


    # initialize OpenCV's special multi-object tracker
    trackers = cv2.legacy.MultiTracker_create()
    #list_of_trackers = []
    #boxes_t = []
    # if a video path was not supplied, grab the reference to the web cam
    cap = cv2.VideoCapture("../../DJI_0148.mp4")

    frame_id = 1
    #cap.set(1, frame_id-1)
    ret,bg_rgb = cap.read()

    Fix_obj = FixView(bg_rgb)
    BG_s = BG_substractor(bg_rgb)

    ret, frame = cap.read()

    fg_img= BG_s.bg_substract(frame)
    objs_taken = False
    Tracking_op = None
    # loop over frames from the video stream
    while True:
        # grab the current frame, then handle if we are using a
        # VideoStream or VideoCapture object
        if frame is None:
            break
        if not objs_taken:
            frame = Fix_obj.fix_view(frame,fg_img)

            fg_img = BG_s.bg_substract(frame)
        else:
            #must be fixed before
            #just for testing
            Tracking_op.track(frame,frame_id)
            frame = Tracking_op.draw(frame)


        # check to see if we have reached the end of the stream


        # resize the frame (so we can process it faster)
        #frame = imutils.resize(frame, width=600)

        # grab the updated bounding box coordinates (if any) for each
        # object that is being tracked

        #draw


        """
        for i,t in enumerate(list_of_trackers):
            state, box_t = t.update(frame)
            #print(state)
            if state:
                boxes_t[i] = box_t
                (x, y, w, h) = [int(v) for v in box_t]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)
            else:
                # faild: draw in red
                (x, y, w, h) = [int(v) for v in boxes_t[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=4)
        """


        #(success, boxes) = trackers.update(frame)
        #print(success)
        # loop over the bounding boxes and draw then on the frame
        #for box in boxes:
        #    (x, y, w, h) = [int(v) for v in box]
        #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)

        # show the output frame
        cv2.imshow("Frame", resize(frame,0.2))
        key = cv2.waitKey(20) & 0xFF

        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if frame_id==4:
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            _, objs_list = BG_s.get_big_objects(fg_img,frame)
            Tracking_op = TrackerObj(frame,objs_list,type="kcf")
            #for r in objs_list:
            #    box = [r.bbox[1],r.bbox[0],r.bbox[3]-r.bbox[1],r.bbox[2]-r.bbox[0]]
            #    if all(box):
            #        boxes_t.append(box)
            #        list_of_trackers.append(cv2.TrackerKCF_create())
            #        list_of_trackers[-1].init(frame,box)

            # create a new object tracker for the bounding box and add it
            # to our multi-object tracker
                #tracker = OPENCV_OBJECT_TRACKERS["mosse"]()
                # trackers.add(tracker, frame, box)
            objs_taken = True

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
        frame_id += 1
        ret, frame = cap.read()
        print(frame_id)



    cap.release()

    # close all windows
    cv2.destroyAllWindows()