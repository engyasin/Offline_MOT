import cv2
import numpy as np
from config import config

from utils import test_box

class TrafficObj():

    def __init__(self,frame,frame_id,box,track_id,tracker=cv2.TrackerKCF_create,class_id=-1,detection_way=1,detect_prob=0.0):
        # -1 for class id means unkonwn

        # detection way:
        # 1 for detection
        # 2 tracking
        # 3 bg substract

        ## Tracking part
        # box is : x,y,w,h

        self.box = box
        self.boxes = [box]
        self.true_wh_max = box[2:],1
        self.tracking_state = [1]
        self.checked_bg = []

        self.time_steps = [frame_id]
        self.trust_level = [[0,0,0]]
        self.trust_level[0][detection_way-1] = 1
        #TODO add another list for detection confirm
        self.tracker_class = tracker
        self.tracker = self.tracker_class()
        # cv2.TrackerKCF_create())
        self.tracker.init(frame,box)
        self.class_id = class_id

        # length is not standard (class,prob)
        self.class_ids = {-1:0,1:0,2:0,3:0}
        self.class_ids[class_id] += (1/(1-detect_prob))

        self.colors_map = [ (0,0,255), # error
                            (0,255,0), # ped
                            (255,0,0), # cyclist
                            (0, 0, 0), # cars
                            (255,255,255)] # unknown

        self.track_id = track_id

        self.last_detect_prob = detect_prob

        self.img_wh = frame.shape[:2][::-1]

        # to be used in postprocess
        self.angels  = []
        self.centers = []

    def find_center(self):
        return int(self.box[0]+(self.box[2]/2)),int(self.box[1]+(self.box[3]/2))

    def track(self,new_frame,frame_id):
        state, box = self.tracker.update(new_frame)
        #print(state)
        if state:
            #update boxes list
            self.box = box
            self.find_true_size(box)
            self.boxes.append(list(box))

        self.tracking_state.append(state)
        self.time_steps.append(frame_id)
        self.trust_level.append([0,int(state),0])

    def re_init_tracker(self,frame):
        # tracker may have errors
        # re init with detections
        # done after filter by detection
        #print(self.box)
        self.tracker = self.tracker_class()
        self.tracker.init(frame,self.box)


    def draw(self,new_frame):

        # object that is being tracked
        # TODO five color code (3 classes, unknown and error) 
        color_code = self.colors_map[self.class_id*self.tracking_state[-1]]
        #color_code = [(1, 0, 255),(0, 255, 0)][self.tracking_state[-1]]

        (x, y, w, h) = [int(v) for v in self.box]


        ###################### Draw rotated
        """
        if self.theta[-1][0] == self.time_steps[-1] and self.track_id>-1:
            center = int(self.box[0]+(self.box[2]/2)),int(self.box[1]+(self.box[3]/2))
            #print(dims)
            rect = cv2.boxPoints((center,(w,h),-1*np.rad2deg(self.theta[-1][1])))
            rect = np.intp(rect)
            cv2.drawContours(new_frame, [rect], 0, color_code,thickness=4)
            #cv2.drawContours(stabilized_frame, [rect], 0, (255,0,0),4)       
        """

        ################################

        if self.track_id>-1:
            cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=color_code, thickness=4)

            cv2.putText(new_frame,str(self.track_id),(x,y),2,3,color_code,thickness=4)
        else:
            # moving obj
            cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=color_code, thickness=4)


        return new_frame


    def update(self):

        # history length
        under_prosses = len(self.trust_level)<config.min_history

        if not(under_prosses):
            # min detection
            if self.class_id == -1:
                # after some history and still no detection --> delete
                return False,False

            traj_state,sum_traj_state = [] , 0
            for state in self.trust_level:
                traj_state.append(any(state))
                sum_traj_state += sum(state)

            #if object is still, (detection error) or not much detections
            if sum_traj_state<(config.min_history+5):
                # at least five times movement or deletsion
                return False,False
            elif all(traj_state):
                # keep tracking if detected enough
                return True,False
            elif all(traj_state[:config.min_history]):
                # error last, save rest
                return False,True
            else:
                # error in first steps , delete all
                return False,False
                #save only history true longer than n_history
        return True, False
        ##Return: Track, Save

    def good_enough(self):

        #TODO when saved after, the last mistakes in detection should be deleted
        return (config.min_track_thresh*2)<(sum(self.tracking_state)+sum(self.checked_bg))


    def filter_by_detections(self,detections):
        # detections : (p1,p2,prob,class_id)
        center = self.find_center()
        #print(center)
        remaining_detections = detections[:]
        r = 0#-10 # error margin
        for i,obj_item in enumerate(detections):
            #print(obj_item[0],obj_item[1])
            if obj_item[2]>config.detect_thresh:
                condition = (obj_item[0][0]+r <center[0]< obj_item[1][0]-r) * (obj_item[0][1]+r <center[1]< obj_item[1][1]-r)
                if condition :
                    box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]
                    if self.class_id == -1:
                        # unknown type
                        self.class_id = obj_item[3]
                        self.box = box
                        self.boxes.append(box)
                        _ = remaining_detections.pop(i)
                        return True,remaining_detections
                    elif self.class_id == obj_item[3]:
                        # known type
                        self.box = box
                        self.boxes.append(box)
                        _ = remaining_detections.pop(i)

                        return True, remaining_detections
                    #else:
                    #    continue
                    #    return False, remaining_detections

                #output.append(TrafficObj(frame,box,class_id=obj_item[3]))
        
        return False,remaining_detections

    def checked_with_bg(self,result):
        self.checked_bg.append(result)


    def filter_by_detections_dist(self, detections, check=False):

        obj_cntr = self.find_center()
        detections_dists = []

        for obj_item in detections:

            if obj_item[2]<config.detect_thresh:
                detections_dists.append(1e9)
                continue

            dist = np.linalg.norm(((obj_item[0][0]+obj_item[1][0])//2 - obj_cntr[0],
                                    (obj_item[0][1]+obj_item[1][1])//2 - obj_cntr[1]))

            #if (obj_item[3] not in self.class_ids) and (self.class_id != -1):
            #    if dist < config.dist_thresh: self.class_ids.append(obj_item[3])
            #    detections_dists.append(1e9)
            #    continue

            detections_dists.append(dist)

        detections_dists.append(1e9)
        Ok = min(detections_dists) < config.dist_thresh
        #if self.track_id == 4:
        #    print(detections_dists)
        #    print([ob[2] for ob in detections])
        #    print([ob[3] for ob in detections])
        if Ok:
            obj_item = detections.pop(np.argmin(detections_dists))
            box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]

            self.find_true_size(box)
            if check: 
                self.boxes[-1] = box
            else:
                self.boxes.append(box)
            if self.class_id == -1: 
                print('now is turned to ',obj_item[3])
            self.class_id = obj_item[3]
            self.box = box
            self.class_ids[obj_item[3]] += (1/(1-obj_item[2]))
            self.last_detect_prob = obj_item[2]

            #self.need_redetect = not(test_box(box,self.img_wh))

        return Ok, detections 


    def filter_by_bg_objs(self, bg_objs):

        obj_cntr = self.find_center()
        detections_dists = []

        for obj_item in bg_objs:

            #if (obj_item[3] != self.class_id) and (self.class_id != -1):
            #    detections_dists.append(1e9)
            #
            #    continue

            dist = np.linalg.norm((obj_item.centroid[1] - obj_cntr[0],
                                   obj_item.centroid[0] - obj_cntr[1]))
            detections_dists.append(dist)

        Ok = False
        if bg_objs:
            Ok = min(detections_dists) < (config.dist_thresh-12)
        if Ok:

            obj_item = bg_objs.pop(np.argmin(detections_dists))
            # no need to assign boxes, unless
            if not(self.tracking_state[-1]): # and not detected

                box = [obj_item.bbox[1],obj_item.bbox[0],obj_item.bbox[3]-obj_item.bbox[1],obj_item.bbox[2]-obj_item.bbox[0]]

                #box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]
                self.box = box
                self.find_true_size(box)
                self.boxes.append(box)

            #if self.class_id == -1: 
            #    self.class_id = obj_item[3]

        return Ok, bg_objs 

    def set_detection(self,ok):
        self.trust_level[-1][0] = int(ok)

    def set_bg_substract(self,ok):
        self.trust_level[-1][2] = int(ok)


    def set_track_id(self,id_):
        if self.track_id == -1:
            self.track_id = id_


    def find_true_size(self,new_box):
        # if the view is bird-view sizes should be fixed
        # box is : x,y,w,h
        if test_box(new_box,self.img_wh):
            center = self.find_center()
            new_center = int(new_box[0]+(new_box[2]/2)),int(new_box[1]+(new_box[3]/2))
            d_x,d_y = center[0]-new_center[0],center[1]-new_center[1]

            # max difference between d_x,d_y is needed, means angels near 0,90,180,270,360
            if abs(d_x-d_y)>self.true_wh_max[1]:
                self.true_wh_max = new_box[2:],abs(d_x-d_y)