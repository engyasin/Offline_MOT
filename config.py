
class config:

    ### general paprmters
    draw = True
    min_track_thresh = 8

    detect_every_N = 7

    ### background substractor parameters
    bgs_history = 3
    bgs_threshold = 100
    bgs_shadows = False
    bgs_learning = 0.5
    bgs_erosion_size = 3
    bgs_min_area = 850
    # bigger would give boxes near the detected boxes with yolo
    bgs_broder_margin = 12

    ### fix view paramers
    do_fix = False
    fixing_dilation = 13
    min_matches     = 15


    ### Tracking parametrs



    ### Detection paramters
    detect_thresh = 0.35 #Yolo detection
    bg_detect_thresh = 0.2
    # distance to the nearst match between detection and tracking
    dist_thresh = 35 #pixels

    # Filtering Objects:
    min_history = 50

    overlap_thresh = 0.5

    # Smoothing
    do_smooth   = True
    window_size = 7
    polydegree  = 3