import cv2
import numpy as np
import logging
from numpy import dot
from scipy.linalg import inv

from config import configs

from utils_ import check_box
from filterpy.stats import mahalanobis


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
        list of frames where object is tested for detection or tracking.
    trust_level : list
        list of three values rows where each value refer to a boolean,
        indicating the sucess of detection, tracking and background
        subtraction respectively
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
        save the result of the current background subtraction.

    set_track_id(id_)
        set track id for the object once it has a detected class

    find_true_size(new_box)
        Test evey box of the current position whether it moves
        in the horizontal or vertical direction or the closest
        to that and save the current size if so.

    get_detection_format()
        Get the bounding box in (top-left point, bottom-right) point
        format and add class_id and dummy probabilty.

    """

    def __init__(self,frame,frame_id,box,track_id,config=configs(),tracker=cv2.TrackerKCF_create,class_id=-1,detection_way=1,detect_prob=0.0):
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
        cfg : config instance 
            A class instance of all the configuration parameters
        tracker : function
            The builder function for the tracker object 
            (default is cv2.TrackerKCF_create)
        class_id : int, optional
            The class type of the object,unknown, pedestrain, cyclist or 
            car (default is -1)
        detection_way : int, optional
            The way the object is detected, detection network, 
            tracking or background subtraction (default is 1)
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

        self.box = tuple(box)
        self.boxes = [box]
        self.vel = [0,0]
        self.use_kalman = config.use_kalman
        self.cfg = config
        self.class_id = class_id
        
        if self.use_kalman:
            self.init_kalman_matrices()

            self.covs = [self.P.diagonal()[:4].mean()]

        self.true_wh_max = box[2:],1
        self.wh = box[2:]
        self.tracking_state = [1]


        self.time_steps = [frame_id]
        self.trust_level = [[0,0,0]]
        if detect_prob>1:
            #manual
            self.trust_level = [[1,1,1]]
        self.trust_level[0][detection_way-1] = 1
        #TODO add another list for detection confirm
        self.tracker_class = tracker

        if class_id  == -1:
            self.tracker =  cv2.TrackerKCF_create()
        else:
            self.tracker = self.tracker_class()

        self.tracker.init(frame,self.box)
        

        # length is not standard (class,prob)
        self.class_ids = {-1:0,1:0,2:0,3:0}
        self.class_ids[class_id] += (1/(1-detect_prob-(1e-5)))

        # error + color_code + unknown
        self.colors_map = [(0,0,255)] + self.cfg.colors_map + [(255,255,255)] 

        self.track_id = track_id

        self.last_detect_prob = detect_prob

        self.img_wh = frame.shape[:2][::-1]

        # to be used in postprocess
        self.angels  = []
        self.centers = []



    def get_F(self,dt):
        """Get the transition matrix for the kalman filter"""
        F = np.array([[1,0,0,0,dt,0],[0,1,0,0,0,dt],[0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]])

        return F


    def get_R(self,detection_way,detect_prob=0.5,bbox_wh=[50,50]):
        """Get the measurement noise matrix for the kalman filter"""
        R = np.zeros((4,4))
        noise_cov = [self.cfg.detection_min_var/(1+detect_prob),self.cfg.tracking_var,
                        [self.cfg.bgs_var,self.cfg.bgs_var]][detection_way-1]#bbox_wh[0]/20,bbox_wh[1]/20
        if type(noise_cov) == list:
            R[0,0] = (noise_cov[0]**2)
            R[1,1] = (noise_cov[1]**2)
            R[2,2] = (noise_cov[0]**2)
            R[3,3] = (noise_cov[1]**2)
        else:
            R[0,0] = noise_cov**2
            R[1,1] = noise_cov**2
            R[2,2] = noise_cov**2
            R[3,3] = noise_cov**2

        return R


    def init_kalman_matrices(self) -> None:
        """Initialize the kalman filter for the object"""
        if self.class_id in [2,3]:
            self.process_cov = self.cfg.process_var*self.class_id
        else:
            self.process_cov = self.cfg.process_var*self.class_id
        process_var = (self.process_cov)**2
        initial_cov = int(self.process_cov*1.4)**2

        self.Q = np.zeros((6,6))
        line_q1 = np.array([process_var,0,process_var,0,process_var,0])
        line_q2 = np.array([0,process_var,0,process_var,0,process_var])
        self.Q[4] = line_q1
        self.Q[5] = line_q2
        self.Q[:,4] = line_q1
        self.Q[:,5] = line_q2


        self.H = np.hstack((np.eye(4),np.zeros((4,2))))

        self.P = np.eye(6)*initial_cov
        self.x = self.get_state()

    def predict_box(self,frame_id):
        """Predict with fixed speed the next bounding box
        """

        dt = frame_id - self.time_steps[-1]
        F = self.get_F(dt)
        x = self.get_state()

        #self.box_ = []#new_box
        #self.boxes_.append(new_box)
        #self.covs.append(way)

        self.x = dot(F, x)
        self.P = dot(F, self.P).dot(F.T) + self.Q

        self.trust_level.append([0,0,0])

    def get_state(self,box=[]):

        state = list(self.box[:2])
        state.extend([self.box[0]+self.box[2],self.box[1]+self.box[3]])

        if box:
            state = list(box[:2])
            state.extend([box[0]+box[2],box[1]+box[3]])

        state.extend(self.vel)
        
        return np.array([state]).T
    
    def set_box(self):
        """Set the box from the updated state"""
        
        self.covs.append(self.P.diagonal()[:4].mean())


        state = self.x[:4].T[0].copy()
        state[2:] = ( state[2:] - state[:2] )
        box = state.astype(int).tolist()


        #min_size=50
        #e_r = 1.1
        #if (box[2]+box[3])<min_size:
        #    print('enlarging')
        #    self.x[0] = max(box[0]-box[2]*(e_r-1),0)
        #    self.x[1] = max(box[1]-box[3]*(e_r-1),0)
        #    self.x[2] = max(box[0]+(e_r*box[2]),0)
        #    self.x[3] = max(box[1]+(e_r*box[3]),0)
        #    state = self.x[:4].T[0].copy()
        #    state[2:] = ( state[2:] - state[:2] )
        #    box = state.astype(int).tolist()
        #    box[2] += min_size
        #    box[3] += min_size

        self.box = box

        # limit vel changes [-100,100]
        self.x[4:] = np.clip(self.x[4:],-1*self.cfg.clip_speed,self.cfg.clip_speed)

        self.vel = self.x[4:].T[0].tolist()

        if len(self.boxes) == len(self.time_steps):
            #replace
            self.boxes[-1] = self.box
        else:
            #add
            self.boxes.append(self.box)

    def update_box(self,new_box,frame_id,way=1):
        """The new box found via:
            1- detection
            2- tracking
            3- background
        """
        #if way in [2,3]:
        #    return None

        #if not((np.array(new_box) - np.array(self.box)).any()):
        #    pass
            #return None
        z = self.get_state(box=new_box)[:4]

        try:
            m_dist = mahalanobis(x=z, mean=self.x[:4], cov=self.P[:4,:4])
        except ValueError:
            Warning('varaince is negative. Something went wrong!')
            return None

        if m_dist > self.cfg.mahalanobis_dist and way !=2:
            return None
        #print('mahalanobis distance = {:.1f}'.format(m_dist))

        if frame_id not in self.time_steps:
            #print(f'frame: {frame_id} ,covarince now {self.P.diagonal().mean()}')
            self.predict_box(frame_id)
            self.time_steps.append(frame_id)

        R = self.get_R(way,self.last_detect_prob,new_box[2:])

        S = dot(self.H, self.P).dot(self.H.T) + R
        K = dot(self.P, self.H.T).dot(inv(S))

        y = z - dot(self.H, self.x)
        #print(f'error vector mean {y.mean()}')
        #if abs(y).mean()>1:
        if dot(K,y)[:4].sum()>400:
            print(f'something fishy with object: {self.track_id}')
            #breakpoint()
        self.x = self.x + dot(K, y)

        self.P = self.P - dot(K, self.H).dot(self.P)



        #if way ==1:
        #    self.wh = new_box[2:]
        # fix x
        #center = (self.x[0]+self.x[2])//2,(self.x[1]+self.x[3])//2

        #self.x[0] = (center[0] - self.wh[0]//2)
        #self.x[2] = (center[0] + self.wh[0]//2)
        #self.x[1] = (center[1] - self.wh[1]//2)
        #self.x[3] = (center[1] + self.wh[1]//2)

        self.set_box()
        self.trust_level[-1][way-1]=1
        #print(self.P.diagonal())
        #self.covs.append(way)


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

        #new_frame_ = new_frame.copy()
        #if self.cfg.Tracker_goturn:
        #    times_ = 2 # each side
        #    c,r,w,h = tuple(self.box)
            #max_r,max_c,_ = new_frame.shape
            #print(c,r,w,h)
        #    new_frame_ = new_frame[min(r-h*times_,0):(r+h+h*times_),min(c-w*times_,0):(c+w+w*times_),:]

        state, box = self.tracker.update(new_frame)

        min_area = 50*50
        re_init = False
        e_r = 1.15
        box = list(box)
        if (box[2]*box[3])<(min_area):
            
            #box[0] = max(box[0]-min_size//2,0)
            #box[1] = max(box[1]-min_size//2,0)
            #box[2] += min_size
            #box[3] += min_size
            box[0] -= box[2]*((e_r-1)/2)
            box[1] -= box[3]*((e_r-1)/2)
            box[2] *= e_r
            box[3] *= e_r

            re_init = True
        #NOTE enlarge by ration
        else:
            e_r = 1.05 #enlarge ratio
            box[0] -= box[2]*((e_r-1)/2)
            box[1] -= box[3]*((e_r-1)/2)
            box[2] *= e_r
            box[3] *= e_r

        # TODO should we add reinint here?


        #if self.cfg.Tracker_goturn:
        #    box = list(box)
        #    box[0] += min(c-w*times_,0)
        #    box[1] += min(r-h*times_,0)
        #    box = tuple(box)
        #print(state)

        box = tuple(box)

        if state:
            #update boxes list
            if self.use_kalman:
                self.update_box(box,frame_id,2)
            else:
                self.box = box
                self.boxes.append(list(box))
                self.time_steps.append(frame_id)
                self.trust_level.append([0,int(state),0])
                
            self.find_true_size(self.box)

        
        self.tracking_state.append(state)
        
        
        if re_init: self.re_init_tracker(new_frame)

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
        self.tracker.init(frame,tuple([int(v)+1 for v in self.box]))

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

            cv2.putText(new_frame,str(self.track_id),(x,y),2,3,color_code,thickness=2)
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
        under_prosses = len(self.trust_level)<self.cfg.min_history

        #if (self.box[2]*self.box[3])<30:
        #    return False

        if not(under_prosses):
            # min detection
            #if self.class_id == -1:
                # after some history and still no detection --> delete
            #    return False,False

            traj_state,sum_traj_state = [] , 0
            for state in self.trust_level:
                traj_state.append(any(state))
                sum_traj_state += sum(state)

            #if object is still, (detection error) or not much detections
            if sum_traj_state<(self.cfg.min_history):
                # at least three time movement or detection
                logging.info(f'Object {self.track_id} deleted because of missing matching steps')
                return False,False
            elif all(traj_state):
                # keep tracking if detected enough
                return True and (self.box[2]*self.box[3])>30,False
            elif (sum(traj_state)/len(traj_state))<self.cfg.missing_thresh:
                #delete
                logging.info(f'Object {self.track_id} deleted because of low detection rate')
                #print(sum(traj_state),len(traj_state))
                return False,True
            #else:
                # error in first steps , delete all
            #    return False,False
                #save only history true longer than n_history
        return True and ((int(self.box[2])*int(self.box[3]))>100), False
        ##Return: Track, Save


    def filter_by_detections_dist(self, detections, check=False):
        """Assign the object to one of the detections in the frame
        if a minimum distance and detection probability are found.

        Parameters
        ----------
        detections : list
            A list for the detections in the current frame as
            it is the output from the detection network
        check : bool, optional
            A flag used to add the result as new point or just update
            the last position (default False)

        Returns
        -------
        tuple
            a tuple of two elements, where the first is boolean whether
            a match of the detection are found. The second is the 
            detection list after removing the matched detection.

        """
        obj_cntr = self.find_center()
        detections_dists = []
        detections_size = []

        for obj_item in detections:

            if obj_item[2]<self.cfg.detect_thresh:
                detections_dists.append(1e9)
                detections_size.append(1e9)
                continue

            dist = np.linalg.norm(((obj_item[0][0]+obj_item[1][0])//2 - obj_cntr[0],
                                    (obj_item[0][1]+obj_item[1][1])//2 - obj_cntr[1]))

            size_ = np.linalg.norm(((obj_item[1][0]-obj_item[0][0]) - self.box[2],
                                   (obj_item[1][1]-obj_item[0][1]) - self.box[3]))

            #if (obj_item[3] not in self.class_ids) and (self.class_id != -1):
            #    if dist < config.dist_thresh: self.class_ids.append(obj_item[3])
            #    detections_dists.append(1e9)
            #    continue

            # if manaual
            if obj_item[2]>1.1:
                dist = (dist/2.5)
                size_ = (size_/2.5)

            detections_dists.append(dist)
            detections_size.append(size_)

        detections_dists.append(1e9)
        detections_size.append(1e9)
        Ok = min(detections_dists) <(self.cfg.dist_thresh)#  (np.sqrt(self.covs[-1])*3)#)
        Ok *= (min(detections_size) < (self.cfg.size_thresh* [1.0,1.35][self.class_id==-1]))
        #if self.track_id == 4:
        #    print(detections_dists)
        #    print([ob[2] for ob in detections])
        #    print([ob[3] for ob in detections])
        if Ok:
            obj_item = detections.pop(np.argmin(detections_dists))
            box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]

            self.find_true_size(box)
            self.last_detect_prob = obj_item[2]
            if check: 
                if self.use_kalman:
                    self.update_box(box,self.time_steps[-1],1)
                self.boxes[-1] = box
            else:
                if self.use_kalman:
                    self.update_box(box,self.time_steps[-1],1)
                else:
                    self.boxes.append(box)
            if self.class_id == -1: 
                logging.info(f'Object {self.track_id} now is turned to {obj_item[3]}')

            self.class_id = obj_item[3]

            if not(self.use_kalman): self.box = box
            self.class_ids[obj_item[3]] += (1/(1-obj_item[2]))

            #self.need_redetect = not(check_box(box,self.img_wh))

        return Ok, detections 

    def change_tracker(self,frame):
        """Change the tracker type of the object

        Parameters
        ----------
        frame : numpy.ndarray
            The current frame of the video

        """
        self.tracker = self.tracker_class()
        self.tracker.init(frame, self.box)


    def filter_by_bg_objs(self, bg_objs):
        """Check if the object is within a minimum distance
        of any moving area.

        Parameters
        ----------
        bg_objs : list
            The list of moving object in the current frame, as
            regions class instances of skimage library

        Returns
        -------
        tuple
            a tuple of two elements, where the first is boolean whether a
            match with one of the moving objects are found. The second is
            the list of moving objects after removing the matched object

        """
        obj_cntr = self.find_center()
        detections_dists = []
        detections_size = []

        for obj_item in bg_objs:

            #if (obj_item[3] != self.class_id) and (self.class_id != -1):
            #    detections_dists.append(1e9)
            #
            #    continue

            dist = np.linalg.norm((obj_item.centroid[1] - obj_cntr[0],
                                   obj_item.centroid[0] - obj_cntr[1]))

            size_ = np.linalg.norm(((obj_item.bbox[3]-obj_item.bbox[1]) - self.box[2],
                                   (obj_item.bbox[2]-obj_item.bbox[0]) - self.box[3]))

            detections_dists.append(dist)
            detections_size.append(size_)

        Ok = False
        if bg_objs:
            ##(np.sqrt(self.covs[-1])*3))
            Ok = (min(detections_dists) <(self.cfg.dist_thresh-12)) *(min(detections_size) < (self.cfg.size_thresh-12))
        if Ok:

            obj_item = bg_objs.pop(np.argmin(detections_dists))
            # no need to assign boxes, unless
            if not(self.tracking_state[-1]): # and not detected

                box = [obj_item.bbox[1],obj_item.bbox[0],abs(obj_item.bbox[3]-obj_item.bbox[1]),abs(obj_item.bbox[2]-obj_item.bbox[0])]

                #box = [obj_item[0][0], obj_item[0][1], obj_item[1][0]-obj_item[0][0],obj_item[1][1]-obj_item[0][1]]
                if self.use_kalman:
                    self.update_box(box,self.time_steps[-1],3)
                else:
                    self.box = tuple(box)
                    self.boxes.append(box)

                self.find_true_size(box)

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
        """save the result of the current background subtraction.

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
        if check_box(new_box,self.img_wh):
            center = self.find_center()
            new_center = int(new_box[0]+(new_box[2]/2)),int(new_box[1]+(new_box[3]/2))
            d_x,d_y = center[0]-new_center[0],center[1]-new_center[1]

            # max difference between d_x,d_y is needed, means angels near 0,90,180,270,360
            if abs(d_x-d_y)>self.true_wh_max[1]:
                self.true_wh_max = new_box[2:],abs(d_x-d_y)

    def get_detection_format(self):
        """Get the bounding box in (top-left point, bottom-right point)
        format and add class_id and dummy probabilty to get a similar
        format to the output of Yolo

        """

        return [(self.box[0],self.box[1]),
                (self.box[0]+self.box[2],self.box[1]+self.box[3]),
                0.5,self.class_id]
