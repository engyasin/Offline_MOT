
class config:

    """
    A class used to set parameters for offline traffic objects detection
    and tracking from drone bird-eye videos.

    ...

    Attributes
    ----------
    draw : boolean
        Whether to show tracking results as it's calculated 
        (default True)
    detect_every_N : int
        The frequency of the detection with Yolov4 network in the
        video. (default 7)
    use_cuda : boolean
        Whether to perform detection on GPU or CPU (default False)
    bgs_history : int
        The number of frames needed for background substractor
        before esitimating the background (default 3)
    bgs_threshold : int
        The distance from which two objects are considered one
        in background substraction result (default 100)
    bgs_shadows : boolean
        Detect the shadows in a different color grade 
        in background substraction (default False)
    bgs_learning : float
        The rate which the background is learned 
        in background substraction (default 0.5)
    bgs_erosion_size : int
        The kernel size to do erosion with on the
        background substraction result (default 3)
    bgs_min_area : int
        The minimum area in pixel for the objects in the result of
        background substraction (default 850)
    bgs_broder_margin : int
        The margin around the objects already detected from which
        overlapping with other objects from background substraction
        is ok (default 12)
    do_fix : boolean
        Whether to perform fixing the view or not (default False)
    fixing_dilation : int
        The kernel size to do dilation to a mask from which matching with
        a reference image will be done for esitimating the transformation
        (default 13)
    min_matches : int
        The minimum number of matches to transform the image according to
        a reference image postion (default 15)
    detect_thresh : float
        The minimum probability to consider a detection result from Yolo
        model ok. (default 0.35)
    dist_thresh : int
        The maximum distance between tracked object position
        and a detected position to consider a correction 
        for its position (default 35)
    min_history : int
        The smallest time steps length that could be saved as a
        traffic object in the result (default 50)
    overlap_thresh : float
        The minimum percentage of objects area overlapping with another
        to choose one of them as duplicate and delete it (default 0.5)
    do_smooth : boolean
        Whether to smooth the tracking trajectories points
        according to Savitzky-Golay algorithm. (default True)
    window_size : int
        The window size for the smoothing algorithm (default 7)
    polydegree : int
        The polydegree for the smoothin algorithm, it must be
        smaller than window_size (default 3)
    """

    ### general paprmters
    draw = True
    detect_every_N = 7
    use_cuda = False

    ### background substractor parameters
    bgs_history = 3
    bgs_threshold = 100
    bgs_shadows = False
    bgs_learning = 0.5
    bgs_erosion_size = 3
    bgs_min_area = 850
    bgs_broder_margin = 12     # bigger would give boxes near the detected boxes with yolo


    ### fix view paramers
    do_fix = False
    fixing_dilation = 13
    min_matches     = 15


    ### Detection paramters
    detect_thresh = 0.35 #Yolo detection
    # distance to the nearst match between detection and tracking
    # output in pixels
    dist_thresh = 35 


    ### Filtering Objects:
    min_history = 50
    overlap_thresh = 0.5


    ### Smoothing for post processing
    do_smooth   = True
    window_size = 7
    polydegree  = 3