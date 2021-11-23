import cv2
import numpy as np
from config import config

from utils import test_box

class TrafficObj():
    """
    A class used to represent any traffic entity, with its full track 
    across frame until it gets lost.

    ...

    Attributes
    ----------
    box : list
        a box is list of [x,y,w,h] where x,y is the top-left point
        coordinates and w,h are the width and height of the bounding box
        unrotated.
    boxes : list
        a list of box lists for every time step saved in time_steps
    true_wh_max : tuple
        the width and height when the object is moving horizontally,
        vertically or the nearst to these directions
    tracking_state : list
        the list of the tracking success variable
    time_steps : list
        list of frames where object are detected or tracked successfully.
    trust_level : list
        list of three values rows where each value refer to a boolean,
        indicating the sucess of detection, tracking and background
        substraction respectively
    tracker_class : Tracker function
        The function that will build the tracker object (default: TrackerKCF_create)
    tracker : Tracker object
        the tracker object that will perform the tracking
    class_id : int
        an integer representing the type of object whether it is
        -1: unknown (default), 1 pedistrains, 2 cyclist, 3 cars.
    colors_map : list
        what color to draw each traffic object
    track_id : int
        a unique id assigned to each traffic object
    last_detect_prob : float
        the probabilty of the last detection.
        it is needed to solve overlaping conflict (default:0)
    img_wh : tuple
        the width and height of the frame
    angels : list
        a placeholder list to calculate the angle of direction at
        the end for the object.
    centers : list
        a placeholder list to calculate the centers of the boxes
        at the end for the object.

    Methods
    -------
    find_center()
        Calculate the current center for the box. 
        the output is integers

    track(new_frame,frame_id)
        Track the object in the new frame, save the result,
        and update the true size

    re_init_tracker(frame)
        start a new tracker object in the frame provided with
        the current box

    draw(new_frame=numpy array)
        Draw the current box position with color code and track ID
        on the frame

    update()
        Test the object if still need to be tracked, or it
        is lost so if it needs to be deleted

    filter_by_detections_dist(detections,check=False)
        It assigns the object to one of the detections in the frame
        if a minimum distance and detection probability is found.

    filter_by_bg_objs(bg_objs)
        check if the object is within a minimum distance
        to any moving area. 

    set_detection(ok)
        save the result of the current detection. 

    set_bg_substract(ok)
        save the result of the current background substraction.

    set_track_id(id_)
        set track id for the object once it has a detected class

    find_true_size(new_box)
        Test evey box of the current position whether it moves
        in the horizontal or vertical direction or the closest
        to that and save the current size if so.

    """

    def __init__(self,frame,frame_id,box,track_id,tracker=cv2.TrackerKCF_create,class_id=-1,detection_way=1,detect_prob=0.0):
        """
        Parameters
        ----------
        frame : numpy array
            The image of the first occurrence of the object
        frame_id : int
            The frame order in the video
        box : int
            The current bounding box of the object, represented as
            [x,y,w,h], where x,y is the top left corner, and
            w,h are the width and height.
        track_id : int
            The object unique identifier
        tracker : function
            The builder function for the tracker object 
            (default is cv2.TrackerKCF_create)
        class_id : int, optional
            The class type of the object,unknown, pedestrain, cyclist or 
            car (default is -1)
        detection_way : int, optional
            The way the object is detected, detection network, 
            tracking or background substraction (default is 1)
        detect_prob : float, optional
            The probabilty that the network detected the 
            object with (default is 0.0)
        """
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
        """Calculate the current center for the box. 
        the output is integers

        Returns
        -------
        tuple
            a tuple of the center x and y as rounded integers

        """
        return int(self.box[0]+(self.box[2]/2)),int(self.box[1]+(self.box[3]/2))

    def track(self,new_frame,frame_id):
        """Track the object in the new frame, save the result,
        and update the true size

        Parameters
        ----------
        new_frame : numpy array
            The frame to do the tracking in
        frame_id : int
            The current frame order in the video

        """
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
        """start a new tracker object in the provided frame with
        the current box

        Parameters
        ----------
        frame : numpy array
            The current frame to initilize the tracking in
        """
        # tracker may have errors
        # re init with detections
        # done after filter by detection
        #print(self.box)
        self.tracker = self.tracker_class()
        self.tracker.init(frame,self.box)

    def draw(self,new_frame):

        """Draw the current box position with color code and track ID
        on the frame

        Parameters
        ----------
        new_frame : numpy array
            The current frame to draw the object in

        Returns
        -------
        numpy array
            The new frame that should be shown with the object
            data drawn in it.

        """

        # object that is being tracked
        # TODO five color code (3 classes, unknown and error) 
        color_code = self.colors_map[self.class_id*self.tracking_state[-1]]
        #color_code = [(1, 0, 255),(0, 255, 0)][self.tracking_state[-1]]

        (x, y, w, h) = [int(v) for v in self.box]


        ################################

        if self.track_id>-1:
            cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=color_code, thickness=4)

            cv2.putText(new_frame,str(self.track_id),(x,y),2,3,color_code,thickness=4)
        else:
            # moving obj
            cv2.rectangle(new_frame, (x, y), (x + w, y + h), color=color_code, thickness=4)


        return new_frame


    def update(self):
        """Test the object if still need to be tracked, or it
        is lost so if it needs to be deleted

        Returns
        -------
        tuple
            a tuple of two boolean varaible, the first to test if 
            the object are lost and the second to test if it was
            real object or noise.

        """
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


    def filter_by_detections_dist(self, detections, check=False):
        """Assign the object to one of the detections in the frame
        if a minimum distance and detection probability are found.

        Parameters
        ----------
        detections : list
            A list for the detections in the current frame as
            output from the detection network
        check : bool, optional
            A flag used to add the result as new point or just update
            the last position

        Returns
        -------
        tuple
            a tuple of two elements, where the first is boolean whether
            a match of the detection are found. The second is the 
            detection list after removing the matched detection.

        """
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
        """Check if the object is within a minimum distance
        of any moving area.

        Parameters
        ----------
        bg_objs : list
            The list of moving object in the current frame

        Returns
        -------
        tuple
            a tuple of two elements, where the first is boolean whether a
            match with one of the moving objects are found. The second is
            the list of moving objects after removing the matched object

        """
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
        """save the result of the current detection. 

        Parameters
        ----------
        ok : boolean
            Whether detection found or not
        """
        self.trust_level[-1][0] = int(ok)

    def set_bg_substract(self,ok):
        """save the result of the current background substraction.

        Parameters
        ----------
        ok : boolean
            Whether moving object found or not
        """
        self.trust_level[-1][2] = int(ok)


    def set_track_id(self,id_):
        """set track id for the object once it has a detected class

        Parameters
        ----------
        id_ : int
            The id to set for the object if it does not have one yet
        """
        if self.track_id == -1:
            self.track_id = id_


    def find_true_size(self,new_box):
        """Test evey box of the current position whether it moves
        in the horizontal or vertical direction or the closest
        to that and save the current size if so.

        Parameters
        ----------
        new_box : list
            The current box to test its direction of movement
        """
        # if the view is bird-view sizes should be fixed
        # box is : x,y,w,h
        if test_box(new_box,self.img_wh):
            center = self.find_center()
            new_center = int(new_box[0]+(new_box[2]/2)),int(new_box[1]+(new_box[3]/2))
            d_x,d_y = center[0]-new_center[0],center[1]-new_center[1]

            # max difference between d_x,d_y is needed, means angels near 0,90,180,270,360
            if abs(d_x-d_y)>self.true_wh_max[1]:
                self.true_wh_max = new_box[2:],abs(d_x-d_y)