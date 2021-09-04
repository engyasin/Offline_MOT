import numpy as np
import cv2
from skimage.measure import label, regionprops

class Element:
    def __init__(self,frame_id,frame_image,rgn,matched=True):
        self.frame_ids = [frame_id]
        self.frame_img = frame_image
        self.size = [rgn.image.shape]
        self.matching_place = [rgn.centroid]
        self.success = [matched]
        self.dist = [] # between the matching and the move
        self.color = tuple([np.random.randint(256) for _ in range(3)])

    def add_match(self,frame_id,rgn,L):
        self.frame_ids.append(frame_id)
        self.size.append(rgn.image.shape)
        self.matching_place.append(rgn.centroid)
        self.success.append(True)
        self.dist.append(L)

    def add_mismatch(self,frame_id):
        self.frame_ids.append(frame_id)
        self.size.append(self.size[-1])
        self.matching_place.append(self.matching_place[-1])
        self.success.append(False)
        self.dist.append(1000000)

    def get_thresh(self):
        if self.dist :
            return max(self.dist[-1]*50,150)
        else:
            return max(self.size[-1])*50

def Template_p_match(image, template , bbox,mask):
    W = int((bbox[2] -bbox[0])*1.5)
    H = int((bbox[3] -bbox[1])*1.5)
    Nbox = [ max(0,bbox[0]- W) , max(0,bbox[1]- H), bbox[2] + W, bbox[3] + H]
    #print(image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    #print(template.shape)
    #print(image.shape,bbox)
    #template[template == 0] = 127
    N_templ = image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:]
    res = cv2.matchTemplate(N_templ,template,
    cv2.TM_CCORR_NORMED,mask=np.float32(mask))
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    #print("***",max_val)
    #print(res[max_loc[::-1]],max_val)
    #print(max_loc,image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    # test the value?

    return Nbox,max_val,(max_loc[0]+Nbox[1],max_loc[0]+Nbox[0])


def templatematch( template ,image,mask):

    #print(image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    #print(template.shape)
    #print(image.shape,bbox)
    #template[template == 0] = 127
    res = cv2.matchTemplate(image,template,cv2.TM_CCORR_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    return mask[max_loc[::-1]],max_loc[::-1]
    #print("***",max_val)
    #print(res[max_loc[::-1]],max_val)
    #print(max_loc,image[Nbox[0]:Nbox[2]+1,Nbox[1]:Nbox[3]+1,:].shape)
    # test the value?

def test_places(places,n):
    r = 1
    s = places[0]
    for p in places[1:]:
        if np.linalg.norm(np.array(s)-np.array(p)) < 50:
            r += 1
    return r>=n

def assocciate(A,B):
    R = []
    dists = []
    end_ = min(len(A),len(B))
    D = np.linalg.norm(np.array(A[0])-np.array(B),axis=1).T
    for a in A[1:]:
        L = np.linalg.norm(np.array(a)-np.array(B),axis=1).T
        D = np.vstack((D,L))
    D = D.T
    try: num_c = D.shape[1]
    except: num_c = 1
    for _ in range(end_):
        X = np.argmin(D)
        dists.append(D.min())
        r = ((X+1)//num_c)
        c = ((X+1)%num_c)-1
        if c == -1:
            r -= 1
            c = num_c-1
        R.append((c,r))
        try:
            D[:,c] = 10e100
            D[r,:] = 10e100
        except:
            D[X] = 10e100
    return R,dists

def wighted_associate(A,B,SA,SB):
    SD = np.array([[np.sqrt(abs(a-b)) for b in SB] for a in SA]).T
    R = []
    dists = []
    end_ = min(len(A),len(B))
    D = np.linalg.norm(np.array(A[0])-np.array(B),axis=1).T
    for a in A[1:]:
        L = np.linalg.norm(np.array(a)-np.array(B),axis=1).T
        D = np.vstack((D,L))
    D = D.T
    D = D+SD
    try: num_c = D.shape[1]
    except: num_c = 1
    for _ in range(end_):
        X = np.argmin(D)
        dists.append(D.min())
        r = ((X+1)//num_c)
        c = ((X+1)%num_c)-1
        if c == -1:
            r -= 1
            c = num_c-1
        R.append((c,r))
        if len(D.shape)==1:
            D[:] = 10e100
        else:
            D[:,c] = 10e100
            D[r,:] = 10e100
    
    return R,dists

def not_within_frame(frame,C,M=5):
    C = np.array(C)
    A = np.any(C<=M)
    A1 = C[0]>(frame.shape[0]-M)
    A2 = C[1]>(frame.shape[1]-M)
    return A or A1 or A2


def postprocForg(F):

	# find shadows
    F_shadows = (F==127)
    F[F>1] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    F = cv2.erode(F,kernel,iterations = 1)

    label_image = label(F)

    Accepted_ratio = 0.25
    Accepted_extent = 0.38

	# first 4 frames are wrong
    regs_str = regionprops(label_image)
    for region in regs_str:
        a = region.major_axis_length 
        a+= 0.00001
        b = region.minor_axis_length
        shadow_r = 1
        if np.any(F_shadows[label_image==region.label]):
            shadow_r = 0.7
        if((b/a)<(Accepted_ratio*shadow_r)) or region.extent<(Accepted_extent*shadow_r):
			# not an object (line maybe)
            F[region.slice] = 0

    F = cv2.dilate(F,kernel_dilate,iterations = 3)
    return F