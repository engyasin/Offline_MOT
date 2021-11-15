import numpy as np
from scipy.signal import savgol_filter
from config import config

# find centers
# smooth cenetrs (Apply a Savitzky-Golay filter to an array.)
# change the boxes (recalculate top-left corner based on new size and center)
# find thetas


def find_cntrs(boxes):
    # box is : x,y,w,h
    centers = []
    for box in boxes:
        centers.append([(box[0]+(box[2]/2)),(box[1]+(box[3]/2))])
    return centers


def tracks_angels(track):
    # track is a list of (x,y)
    angels = []
    for i,p1 in enumerate(track[:-1]):
        p2 = track[i+1]
        # (y2-y1)/(x2-x1)
        ang = np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0]))
        angels.append(np.rad2deg(ang))
        # in deg
    angels.append(angels[-1])
    return angels

def post_process(obj):

    obj.centers = find_cntrs(obj.boxes)
    # smooth
    if config.do_smooth:
        obj.centers = savgol_filter(obj.centers,config.window_size,config.polydegree,axis=0)
    # change boxes
    w,h = obj.true_wh_max[0][0],obj.true_wh_max[0][1]
    for i,cntr in enumerate(obj.centers):
        obj.boxes[i] = [int(cntr[0]-(w/2)),int(cntr[1]-(h/2)),w,h]
    # find angels
    obj.angels  = tracks_angels(obj.centers)
    #if config.do_smooth:
    #    obj.angels = savgol_filter(obj.angels,config.window_size,config.polydegree,axis=0)
    return obj

def postProcessAll(tracks_objs):
    res = []
    for obj in tracks_objs:
        res.append(post_process(obj))
    return res