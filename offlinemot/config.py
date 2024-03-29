import os,sys
import configparser
class configs:

    """
    A class used to set parameters for offline traffic objects detection
    and tracking from drone bird-eye videos.

    ...

    Attributes
    ----------
    draw : boolean
        Whether to show tracking results as it's calculated 
    detect_every_N : int
        The frequency of the detection with Yolov4 network in the
        video. 
    missing_thresh : float
        Maximum percentage of correct tracking and detection for any 
        object to continue tracking it
    use_cuda : boolean
        Whether to perform detection on GPU or CPU 
    resize_scale: float
        The resizing scale of the image to show while processing.
        1 means the same as the true size
    color_map : list
        list of RGB colors to represent each detected class while
        drawing (red and white are reserved). It depends on number
        of output classes in yolo config
    bgs_history : int
        The number of frames needed for background substractor
        before esitimating the background 
    bgs_threshold : int
        The distance from which two objects are considered one
        in background substraction result 
    bgs_shadows : boolean
        Detect the shadows in a different color grade 
        in background substraction 
    bgs_learning : float
        The rate which the background is learned 
        in background substraction 
    bgs_erosion_size : int
        The kernel size to do erosion with on the
        background substraction result 
    bgs_min_area : int
        The minimum area in pixel for the objects in the result of
        background substraction 
    bgs_broder_margin : float
        The margin around the objects already detected from which
        overlapping with other objects from background substraction
        is ok. This is represented as percentage of the width and height
        of the object.
    do_fix : boolean
        Whether to perform fixing the view or not 
    fixing_dilation : int
        The kernel size to do dilation to a mask from which matching with
        a reference image will be done for esitimating the transformation
    min_matches : int
        The minimum number of matches to transform the image according to
        a reference image postion 
    model_name : str
        The filename of the trained Yolo model
    model_config : str
        The filename of the configuration file of the trained model
    classes_file_name : str
        The filename where the list of output classes are list. normaly,
        there are three types (cyclists, pedestrians and vehicles)
    detect_thresh : float
        The minimum probability to consider a detection result from Yolo
        model ok. 
    dist_thresh : int
        The maximum distance between tracked object position
        and a detected position to consider a correction 
        for its position 
    detect_scale: float
        How much focus on the image for detection, higher values
        would take the whole image, while 0 value would focus on the
        previously detected objects.
    min_history : int
        The smallest time steps length that could be saved as a
        traffic object in the result 
    overlap_thresh : float
        The minimum percentage of objects area overlapping with another
        to choose one of them as duplicate and delete it 
    do_smooth : boolean
        Whether to smooth the tracking trajectories points
        according to Savitzky-Golay algorithm. 
    window_size : int
        The window size for the smoothing algorithm 
    polydegree : int
        The polydegree for the smoothin algorithm, it must be
        smaller than window_size 
    save_out_video : bool
        Whether to save the output video in mp4 format with the
        tracking data overlaid
    """

    ### general parameters
    draw = True
    detect_every_N = 1
    missing_thresh = 0.8
    use_cuda = False
    resize_scale = 0.4
    colors_map = [ (0,255,0), # ped
                   (255,0,0), # cyclist
                   (0, 0, 0)] # cars

    ### background substractor parameters
    bgs_history = 5
    bgs_threshold = 50
    bgs_shadows = True
    bgs_learning = 0.5
    bgs_erosion_size = 3
    bgs_min_area = 300
    bgs_broder_margin =  0.45    # bigger would give boxes near the detected boxes with yolo


    ### fix view paramers
    do_fix = False
    fixing_dilation = 13
    min_matches     = 15


    ### Detection paramters
    cwd = os.path.dirname(os.path.realpath(__file__))
    model_name       = os.path.join(cwd,'model','Yolov4_epoch300.pth')
    model_config     = os.path.join(cwd,'model','yolov4-obj.cfg')
    classes_file_name= os.path.join(cwd,'model','obj.names')

    detect_thresh = 0.3 #Yolo detection
    # distance to the nearst match between detection and tracking
    # output in pixels
    dist_thresh = 25
    size_thresh = 25
    detect_scale = 4.0


    ### Filtering Objects:
    min_history = 100
    overlap_thresh = 0.7


    ### Smoothing for post processing
    do_smooth   = True
    window_size = 7
    polydegree  = 3
    save_out_video = False

    ## Other parameters
    manual_start = True
    double_detection = True
    Tracker_goturn = False
    # [we have a problem ==> push them,
    #  it is that big ==> cut them, it's too much ==> delete the bigger]
    overlap_steps = [0.15,0.33,1.01]


    # kalman parameters
    use_kalman = True
    mahalanobis_dist = 50
    process_var = 90
    bgs_var = 70
    tracking_var = 7
    detection_min_var = 15
    clip_speed = 15

    def __init__(self,file_name=None):

        if file_name is None:
            file_name = os.path.join(self.cwd,'model','config.ini')

        self.config_obj = configparser.ConfigParser()
        self.config_obj.read(file_name)

        self.copy_()

        #self.print_summary()


    def print_summary(self):
        """Show some important parameters
        """
        print('Used Parameters')
        print('====================')
        print('Groupus: ')
        for section in self.config_obj.sections():
            print(section)

        print('====================')
        print('General Parameters: ')
        print('====================')
        for key in self.config_obj['General parameters']:  
            print(':'.join([key,self.config_obj['General parameters'][key]]))


    def __getitem__(self,key):
        for section in self.config_obj.sections():
            if key in self.config_obj[section]:
                print(self.config_obj[section][key])
                break


    def __setitem__(self, key, item):

        for section in self.config_obj.sections():
            if key in self.config_obj[section]:
                break

        setattr(self,key,item)
        if type(item) is not str:
            item = str(item)
        self.config_obj[section][key] = item


        self.print_section(section)


    def print_section(self,section):
        """Show  parameters of a single section
        """
        print(section)
        print('====================')
        for key in self.config_obj[section]:  
            print(':'.join([key,self.config_obj[section][key]]))

    def copy_(self):
        """Copy the data from the ini file to the instance
        """
        for section in self.config_obj.sections():
            for key in self.config_obj[section]:
                if key in ['cwd','model_name','model_config','classes_file_name']:
                    if 'os.path' not in self.config_obj[section][key]:
                        # it is a directory
                        setattr(self,key,self.config_obj[section][key])
                        continue
                setattr(self,key,eval(self.config_obj[section][key]))

    def write(self,filename):
        """Save the current set of parameters values
        """
        with open(filename, 'w') as configfile:
            self.config_obj.write(configfile)

