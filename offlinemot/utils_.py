import cv2
import numpy as np

from config import configs
import os, logging
import gdown


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

    filedir,filename = os.path.split(filename)
    f = open(os.path.join(filedir,filename+'.txt'),mode='w+')
    for obj in tracks_objs:
        class_ = max(obj.class_ids,key=lambda x:obj.class_ids[x])
        #T = np.array(sorted(obj.centers,key=lambda x: x[0]))
        #x,y = T.T[0],T.T[1]
        #spl = splrep(x, y, s=0.2) #Larger s means more smoothing 
        #print(spl)
        w,h = obj.true_wh_max[0][0],obj.true_wh_max[0][1]
        for i,frame_id in enumerate(obj.time_steps):

            # not taking the last step if it's wrong
            if i>= len(obj.boxes) or i>= len(obj.angels):
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
    filedir,filename = os.path.split(filename)
    tracking_data = {}
    with open(os.path.join(filedir,filename+'.txt'),mode='r') as f:
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
    return (box[0]>=0)*(box[1]>=0)*((box[0]+box[2])<img_wh[1])*((box[1]+box[3])<img_wh[0])*(box[2]>=0)*(box[3]>=0)

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

def detect_overlaping(objects,overlap_thresh=0.5,overlap_steps=[0.15,0.33,1.01]):

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

                if (obj.class_id == -1):
                    return i
                elif (other_obj.class_id == -1 ):
                    return j

                overlap_ratio = (area/min((obj.box[2]*obj.box[3]),(other_obj.box[2]*other_obj.box[3])))
                if overlap_ratio >overlap_steps[2]:
                    # NOTE
                    # one inside the other
                    # the smaller is better?
                    if (obj.box[2]*obj.box[3]) < (other_obj.box[2]*other_obj.box[3]):
                        return j
                    elif (obj.box[2]*obj.box[3]) > (other_obj.box[2]*other_obj.box[3]):
                        return i
                elif overlap_ratio >overlap_thresh:
                    # test according to three terms respectivelly
                    # Histroy length, then
                    if len(obj.trust_level)<len(other_obj.trust_level)-1:
                        # longer by two steps
                        return i
                    elif len(obj.trust_level)>len(other_obj.trust_level)+1:
                        return j
                    # detection prob, then
                    elif round(obj.last_detect_prob,2) < round(other_obj.last_detect_prob,2):
                        return i
                    elif round(obj.last_detect_prob,2) > round(other_obj.last_detect_prob,2):
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
                elif overlap_ratio > overlap_steps[0]:
                    #small
                    if (obj.class_id == -1):
                        return i
                    elif (other_obj.class_id == -1 ):
                        return j

                    if overlap_ratio < overlap_steps[1]:
                        obj.box, other_obj.box = find_middle_ground(obj.box,other_obj.box)
                    else:
                        #big. cut them then
                        obj.box, other_obj.box = find_middle_ground2(obj.box,other_obj.box)
                    #print()
                pass
    return -1



def find_middle_ground(box1,box2):

    box1, box2 = list(box1), list(box2)
    # for xs
    x1,x11 = box1[0],box1[0]+box1[2]
    x2,x22 = box2[0],box2[0]+box2[2]

    y1,y11 = box1[1],box1[1]+box1[3]
    y2,y22 = box2[1],box2[1]+box2[3]

    x12 = int((max(x1,x2)+min(x11,x22))/2)
    dx = abs(max(x1,x2)-min(x11,x22))
    y12 = int((max(y1,y2)+min(y11,y22))/2)
    dy = abs(max(y1,y2)-min(y11,y22))

    #if x12<y12 :
    
    if (dx*(box1[3]+box2[3]))<(dy*(box1[2]+box2[2])):

        if x1<x2:
            #box1[2] = x12 - x1

            box2[0] = x12
            #box2[2] = x22 - x12
            box1[0] = x1 - (dx//2)
        else:
            box1[0] = x12
            #box1[2] = x11 - x12

            #box2[2] = x12 - x2
            box2[0] = x2 - (dx//2)

    else:
        if y1<y2:
            #box1[3] = y12 - y1

            box2[1] = y12
            #box2[3] = y22 - y12
            box1[1] = y1 - (dy//2)
        else:
            box1[1] = y12
            #box1[3] = y11 - y12

            #box2[3] = y12 - y2
            box2[1] = y2 - (dy//2)
    
    return tuple(box1) ,tuple(box2)


def find_middle_ground2(box1,box2):

    box1, box2 = list(box1), list(box2)
    # for xs
    x1,x11 = box1[0],box1[0]+box1[2]
    x2,x22 = box2[0],box2[0]+box2[2]

    y1,y11 = box1[1],box1[1]+box1[3]
    y2,y22 = box2[1],box2[1]+box2[3]

    x12 = int((max(x1,x2)+min(x11,x22))/2)
    dx = abs(max(x1,x2)-min(x11,x22))
    y12 = int((max(y1,y2)+min(y11,y22))/2)
    dy = abs(max(y1,y2)-min(y11,y22))

    #if x12<y12 :
    
    if (dx*(box1[3]+box2[3]))<(dy*(box1[2]+box2[2])):

        if x1<x2:
            box1[2] = x12 - x1

            box2[0] = x12
            box2[2] = x22 - x12
            #box1[0] = x1 - (dx//2)
        else:
            box1[0] = x12
            box1[2] = x11 - x12

            box2[2] = x12 - x2
            #box2[0] = x2 - (dx//2)

    else:
        if y1<y2:
            box1[3] = y12 - y1

            box2[1] = y12
            box2[3] = y22 - y12
            #box1[1] = y1 - (dy//2)
        else:
            box1[1] = y12
            box1[3] = y11 - y12

            box2[3] = y12 - y2
            #box2[1] = y2 - (dy//2)
    
    return tuple(box1) ,tuple(box2)



def transform_detection(p0,detections,detect_thresh):
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
    detect_thresh : float
        The minimum probability to consider a detection result from Yolo
        model ok. 

    Returns
    -------
    list
        The same detection list as the input but with moving 
        the coordinates to the original frame of coordinates
        before cropping.
    """

    output = []
    for detection in detections:
        if detection[2]>detect_thresh:
            output.append( [(p0[0]+detection[0][0],p0[1]+detection[0][1]),
                            (p0[0]+detection[1][0],p0[1]+detection[1][1]),
                            detection[2], detection[3]])
    
    return output


def load_model():
    """Download the Yolo network pretrained file for the example for
    the first time
    """
    output = os.path.join(configs.cwd,"model","Yolov4_epoch300.pth")
    if not(os.path.exists(output)) :
        logging.info("Downloading the example pretrained network (Only once)")
        url= "https://drive.google.com/uc?id=1rhDaY7aVSeETP8rHgqZTewp4QkWlr3fb"
        gdown.download(url, output, quiet=False)