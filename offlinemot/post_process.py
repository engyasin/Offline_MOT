import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import splev, splrep

from config import config

# find centers
# smooth cenetrs (Apply a Savitzky-Golay filter to an array.)
# change the boxes (recalculate top-left corner based on new size and center)
# find thetas

def find_cntrs(boxes):
    """Get the centers of the list of boxes in the input.

    Parameters
    ----------
    boxes : list
        The list of the boxes where each box is
        [x,y,width,height] where x,y is the coordinates
        of the top-left point

    Returns
    -------
    list
        Centers of each box where each row is a list of
        two float numbers (x,y).

    """
    # box is : x,y,w,h
    centers = []
    for box in boxes:
        centers.append([(box[0]+(box[2]/2)),(box[1]+(box[3]/2))])
    return centers


def tracks_angels(track):
    """Find the orientation of point that lead to the next one
    in the track input.

    The last orientation is repated to set orientation to the last point.

    Parameters
    ----------
    tracks : list
        The track list of 2D points, representing the centers
        of the tracked target

    Returns
    -------
    list
        a list of angels in degrees related to each point
        in the track in the same order.

    """
    # track is a list of (x,y)
    angels = []
    #change happen on langer time period than 1 frame (1/30 second)
    N = min(10, len(track)-2)
    for i,p1 in enumerate(track[:-1*N]):
        p2 = track[i+N]
        # (y2-y1)/(x2-x1)
        ang = np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0]))
        ang = np.rad2deg(ang)
        angels.append(ang)
        # in deg
    angels.extend([angels[-1]]*N)
    return angels

def repair_traj(obj):
    """Perfrom linear interpolation for a track with missing 
    steps (where there's no detection or tracking).

    Parameters
    ----------
    obj : TrafficObj instance
        The traffic object instance whose track should be interpolated

    Returns
    -------
    list
        a list of the object center coordinates, where the missing
        postions are interpolated lineary using the surrounding positions

    """
    j,i = 0,0
    new_centers = []
    while i<len(obj.trust_level) and j<len(obj.centers):
        if any(obj.trust_level[i]):
            new_centers.append(obj.centers[j])
            j += 1
            i += 1
            continue
        else:
            c = 0
            for state2 in obj.trust_level[i:]:
                if any(state2):
                    break
                else:
                    c+=1
            dx = (obj.centers[j][0]-obj.centers[j-1][0])/(c+1)
            dy = (obj.centers[j][1]-obj.centers[j-1][1])/(c+1)
            i += c
            for x in range(1,c+1):
                new_centers.append((obj.centers[j-1][0]+dx*x,obj.centers[j-1][1]+dy*x))
    return new_centers

def post_process(obj):
    """perform post processing steps on the tracking data.

    The processing contains:
    1) finding and interpolating the centers of boxes
    2) smoothing the track of centers
    3) recalculating the boxes
    4) finding the orientations
    5) smoothing the orientations

    Parameters
    ----------
    obj : TrafficObj class instance
        The target instance which contains the boxes and detection data

    Returns
    -------
    TrafficObj class instance
        The target instance after recalculating and smoothing the 
        orientations and centers

    See Also
    --------
    postProcessAll : loop over all obejcts for post processing.

    """

    obj.centers = find_cntrs(obj.boxes)
    obj.centers = repair_traj(obj)
    # smooth
    if config.do_smooth and len(obj.centers)>=config.window_size:
        obj.centers = savgol_filter(obj.centers,config.window_size,config.polydegree,axis=0)
        #spl = splrep(x, y, s=0.5) #Larger s means more smoothing 
        #y2 = splev(x2, spl)
        # maybe smooth on x also
        
    # change boxes
    w,h = obj.true_wh_max[0][0],obj.true_wh_max[0][1]
    obj.boxes = []
    for i,cntr in enumerate(obj.centers):
        obj.boxes.append([int(cntr[0]-(w/2)),int(cntr[1]-(h/2)),w,h])
    # find angels
    obj.angels  = tracks_angels(obj.centers)
    return obj

def postProcessAll(tracks_objs):
    """It loops over all the objects in the input list for post processing

    Parameters
    ----------
    tracks_objs : list
        The list of all the taregts class objects

    Returns
    -------
    list
        a list of the new post processed tracked objects

    See Also
    --------
    post_process : perform post processing steps on the tracking data. 

    """
    res = []
    for obj in tracks_objs:
        res.append(post_process(obj))
    return res