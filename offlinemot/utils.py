import cv2
import numpy as np

from config import config


def resize(img,scale=1):
    """Resize image by a scale factor 

    Parameters
    ----------
    img : numpy array 
        The image represented as grayscale image array or 
        rgb image array (if grayscale will be stacked to 3 channels)
    scale : float, optional
        A scale to enlarge or shrink the image to both the 
        width and height.  (default is 1)

    Returns
    -------
    numpy array
        The image after rescaling

    """
    if len(img.shape)==2:
        img = np.dstack((img,img,img))

    I_com = cv2.resize(img,
                        tuple([int(x*scale) for x in img.shape[::-1][1:]]))
    return I_com


def save_tracks(tracks_objs,filename):
    """Record the tracking results in a text file,

    It saves the boxes, angels, track ids and classes after
    deleting the parts where no detection, tracking or movement is found,
    and after taking the highst class score of all the detections across 
    the track 

    Parameters
    ----------
    tracks_objs : list
        The list that contains all the objects instances of the tracked 
        tragets 
        that should be saved
    filename : str
        The directory of the video that is under processing. 
        The saved text file will have the same name.

    See Also
    --------
    save_tracks : Record the tracking results in a text file.

    """

    filename = filename.split('\\')[-1].split('.')[-2]#[-5:]
    f = open('outputs\\'+filename+'.txt',mode='w+')
    for obj in tracks_objs:
        class_ = max(obj.class_ids,key=lambda x:obj.class_ids[x])
        #T = np.array(sorted(obj.centers,key=lambda x: x[0]))
        #x,y = T.T[0],T.T[1]
        #spl = splrep(x, y, s=0.2) #Larger s means more smoothing 
        #print(spl)
        w,h = obj.true_wh_max[0][0],obj.true_wh_max[0][1]
        for i,frame_id in enumerate(obj.time_steps):

            # not taking the last step if it's wrong
            if i>= len(obj.boxes):
                break
            # TODO i case class is a miss, maybe flag it with -1 sign
            f.write(' '.join(
                [str(frame_id),str(obj.boxes[i]),str(class_),str(obj.track_id),str(int(obj.angels[i]))])+'\n')

    f.close()


def read_tracks(filename):
    """Read the text file and load it into a dictionary with
    frame number as the keys and the objects and its positions as values.

    Parameters
    ----------
    filename : str
        The directory of the video whose tracking data should
        be shown. The text file will have the same name but '.txt'
        extention

    See Also
    --------
    read_tracks : It loads the tracking data into a dictionary.

    """
    # input : video name
    filename = filename.split('\\')[-1].split('.')[-2]#[-5:]
    tracking_data = {}
    with open('outputs\\'+filename+'.txt',mode='r') as f:
        while True:
            line = f.readline().split()
            if len(line)<1:
                break
            frame_id = int(line[0]) # frame_id
            class_id = int(line[5])
            track_id = int(line[6])
            angel    = int(line[7])
            box = [int(line[1][1:-1]),int(line[2][:-1]),int(line[3][:-1]),int(line[4][:-1])]
            if frame_id in tracking_data:
                # box is x,y,w,h
                tracking_data[frame_id].append((box,class_id,track_id,angel))
            else:
                tracking_data[frame_id] = [(box,class_id,track_id,angel)]

    return tracking_data

def check_box(box,img_wh):
    """ Test whether a box is within image size.

    Parameters
    ----------
    box : list
        A list of [x,y,width,height] where x,y in the top-left 
        point coordinates 

    Returns
    -------
    bool
        A boolean indicating whether the box inside the image 
        dimensions or not.

    """
    return (box[0]>=0)*(box[1]>=0)*((box[0]+box[2])<img_wh[0])*((box[1]+box[3])<img_wh[1])*(box[2]>=0)*(box[3]>=0)

def find_overlap(box1,box2):
    """Find the area of intersection between two boxes

    Parameters
    ----------
    box1 : list
        A list of [x,y,width,height] where x,y in the top-left
        point coordinates of the first box
    box2 : list
        A list of [x,y,width,height] where x,y in the top-left 
        point coordinates of the second box

    Returns
    -------
    int
        The area of the intersection between the two boxes

    Examples
    --------
    >>> find_overlap([0,0,10,5],[0,0,5,10])
     25

    """
    # box is : x,y,w,h
    x1 = set(range(box1[0],box1[0]+box1[2]))
    y1 = set(range(box1[1],box1[1]+box1[3]))

    x2 = set(range(box2[0],box2[0]+box2[2]))
    y2 = set(range(box2[1],box2[1]+box2[3]))

    return len(x1.intersection(x2))*len(y1.intersection(y2))

def detect_overlaping(objects,overlap_thresh=0.5):

    """Check if any object is overlaping within another in the list
    and delete one of them according to: 
    1) history length
    2) detection probability
    3) area
    respectively.

    Parameters
    ----------
    objects : list
        The list of objects instances with boxes attributes
    overlap_thresh : float, optional
        A threshold of accepted overlaping ratio to the object area, 
        before deleting one of the overlapped object (default is 0.5)

    Returns
    -------
    int
        The index of object that should be deleted or -1 if none should

    """

    for i,obj in enumerate(objects):
        for j,other_obj in enumerate(objects):
            if i>=j:# obj.track_id == other_obj.track_id:
                continue
            area = find_overlap(obj.box,other_obj.box)
            if area:
                if (area/min((obj.box[2]*obj.box[3]),(other_obj.box[2]*other_obj.box[3])))>overlap_thresh:
                    # test according to three terms respectivelly
                    # Histroy length, then
                    if len(obj.trust_level)<len(other_obj.trust_level)-1:
                        # longer by two steps
                        return i
                    elif len(obj.trust_level)>len(other_obj.trust_level)+1:
                        return j
                    # detection prob, then
                    elif round(obj.last_detect_prob,1) < round(other_obj.last_detect_prob,1):
                        return i
                    elif round(obj.last_detect_prob,1) > round(other_obj.last_detect_prob,1):
                        return j
                    # area
                    elif (obj.box[2]*obj.box[3]) < (other_obj.box[2]*other_obj.box[3]):
                        return i
                    elif (obj.box[2]*obj.box[3]) > (other_obj.box[2]*other_obj.box[3]):
                        return j
                    else:
                        # choose randomly, they are practicly the same.
                        # to save track id range, choose the minmum
                        return min(i,j)
    return -1



def transform_detection(p0,detections):
    """Convert the result of the detection from a cropped part 
    of image to the original image coordinates.

    Parameters
    ----------
    p0 : tuple
        The top-left point used for cropping the image (x,y)
    detections : list
        A list of lists for the detections in the cropped frame as it is the 
        output from the detection network. The list has the following
        shape [top-left point, bottom-right point,probabilty, class id]

    Returns
    -------
    list
        The same detection list as the input but with moving 
        the coordinates to the original frame of coordinates
        before cropping.
    """

    output = []
    for detection in detections:
        if detection[2]>config.detect_thresh:
            output.append( [(p0[0]+detection[0][0],p0[1]+detection[0][1]),
                            (p0[0]+detection[1][0],p0[1]+detection[1][1]),
                            detection[2], detection[3]])
    
    return output

