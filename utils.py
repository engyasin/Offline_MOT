import cv2
import numpy as np


def resize(img,scale=1):
    if len(img.shape)==2:
        img = np.dstack((img,img,img))

    I_com = cv2.resize(img,
                        tuple([int(x*scale) for x in img.shape[::-1][1:]]))
    return I_com


def save_tracks(tracks_objs,filename):
    filename = filename.split('\\')[-1].split('.')[-2]#[-5:]
    f = open('outputs\\'+filename+'.txt',mode='w+')
    for obj in tracks_objs:
        for i,frame_id in enumerate(obj.time_steps):
            # not taking the last step if it's wrong
            if not(any(obj.trust_level[i])):
                # all zeros
                break
            #TODO add:
            # thetas
            class_ = max(obj.class_ids,key=lambda x:obj.class_ids[x])
            # TODO i case class is a miss, maybe flag it with -1 sign
            f.write(' '.join(
                [str(frame_id),str(obj.boxes[i]),str(class_),str(obj.track_id),str(int(obj.angels[i]))])+'\n')

    f.close()


def read_tracks(filename):
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


def test_box(box,img_wh):
    return (box[0]>=0)*(box[1]>=0)*((box[0]+box[2])<img_wh[0])*((box[1]+box[3])<img_wh[1])*(box[2]>=0)*(box[3]>=0)

def find_overlap(box1,box2):
    # box is : x,y,w,h
    x1 = set(range(box1[0],box1[0]+box1[2]))
    y1 = set(range(box1[1],box1[1]+box1[3]))

    x2 = set(range(box2[0],box2[0]+box2[2]))
    y2 = set(range(box2[1],box2[1]+box2[3]))

    return len(x1.intersection(x2))*len(y1.intersection(y2))

def detect_overlaping(objects,overlap_thresh=0.5):

    for i,obj in enumerate(objects):
        for j,other_obj in enumerate(objects):
            if i>=j:# obj.track_id == other_obj.track_id:
                continue
            area = find_overlap(obj.box,other_obj.box)
            if area:
                if ((obj.box[2]*obj.box[3])/area)>overlap_thresh:
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
                        # this is tricky, maybe leave them for future steps to descid (or at the end)
                        pass
    return -1



