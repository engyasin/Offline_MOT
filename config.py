
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
    bgs_min_area = 750

    ### fix view paramers
    fixing_dilation = 13
    min_matches     = 15


    ### Tracking parametrs



    ### Detection paramters
    detect_thresh = 0.4 #Yolo detection
    bg_detect_thresh = 0.2
    # distance to the nearst match between detection and tracking
    dist_thresh = 35 #pixels

    # Filtering Objects:
    min_history = 50

    overlap_thresh = 0.5