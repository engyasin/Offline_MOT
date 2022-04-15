from tool import darknet2pytorch
import torch

from tool.utils import *
from tool.torch_utils import *

from config import configs
from utils_ import find_overlap, transform_detection, load_model
# from tool.darknet2pytorch import Darknet
import cv2

class YoloDetector():

    """
    A class to load trained Yolov4 model from memory,
    and perform detection on frames.

    ...

    Attributes
    ----------
    m : Pytorch Model class instance 
        A model of Yolov4 architecture
    
    use_cude : bool
        Run on CPU if False or GPU if True

    class_names : list
        list of class names from which detection performed

    Methods
    -------
    detect(numpy array) -> list,(int,int)
        Process a new frame to find the detections along with
        their positions and probabilities

    better_detection(numpy array, list) -> list,(int,int)
        Detect the classes within the image at the proposed areas
        based on the previous detections.

    """

    def __init__(self,config=configs()):

        """
        Parameters
        ----------
        config : config instance 
            A class instance of all the configuration parameters

        """

        cfgfile   = config.model_config
        weightfile=config.model_name
        use_cuda  = config.use_cuda
        namesfile = config.classes_file_name

        self.detect_scale = config.detect_scale
        self.detect_thresh = config.detect_thresh

        self.m = darknet2pytorch.Darknet(cfgfile, inference=True)

        # if GPU not avaliable then CPU
        map_location = torch.device('cpu')
        if torch.cuda.is_available():
            map_location = None

        # if example network loaded for the first time,
        if "Yolov4_epoch300.pth" in weightfile:
            load_model()

        self.m.load_state_dict(torch.load(weightfile,map_location=map_location))
        #self.m.load_weights(weightfile)

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.m.cuda()

        
        self.class_names = load_class_names(namesfile)
    
    def detect(self,imgfile):

        """Detect the classes within the image

        Parameters
        ----------
        imgfile : str or numpy array
            The input image that could be a string filename or a numpy 
            array

        Returns
        -------
        list,(int,int)
            A list of detection positions and classes and the shape of
            the original image. The list has the following shape 
            [top-left point, bottom-right point,probabilty, class id]

        """

        if type(imgfile)==str:
            img = cv2.imread(imgfile)
        else:
            img = imgfile.copy()
        sized = cv2.resize(img, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        # 0.4 is the tresh
        boxes = do_detect(self.m, sized, 0.1, 0.6, self.use_cuda)

        #print(boxes)
        h,w = img.shape[:-1]
        results = []
        for box in boxes[0]:
            p1 = int(box[0]*w) , int(box[1]*h)
            p2 = int(box[2]*w) , int(box[3]*h)
            prob = box[5]
            class_id = box[6] + 1 # 0 reserved to error code, so +1
            results.append((p1,p2,prob,class_id))
            #print('class id: ',class_id)
        #print('**********')

        return results,(w,h)

    def better_detection(self, imgfile, additional_objs=[]):

        """Detect the classes within the image at the proposed areas
        based on the previous detections. This will make yolo network
        take higher resolution input. Additionally the full scale 
        detections are included.

        Parameters
        ----------
        imgfile : str or numpy array
            The input image that could be a string filename or a numpy 
            array
        additional_objs : list
            The list of current tracked and moving objects boxes. These
            will form the cropped parts that the model will perfrom 
            detections on.

        Returns
        -------
        list,(int,int)
            A list of detection positions and classes and the shape of
            the original image. The list has the following shape 
            [top-left point, bottom-right point,probabilty, class id]

        """

        results,(w,h) = self.detect(imgfile)
        results_ = results[:]

        results.extend([obj.get_detection_format() for obj in additional_objs])
        # [top-left point, bottom-right point,probabilty, class id]
        # add margin to each box (equal its max dimination) of all sides
        new_boxes = []
        detections = []
        for box in results:
            max_dim = int(self.detect_scale*max(box[1][0]-box[0][0],box[1][1]-box[0][1]))
            # new box: x,y,w,h
            new_box = [max(box[0][0]-max_dim,0),max(box[0][1]-max_dim,0),
                        box[1][0]-box[0][0]+2*max_dim,
                        box[1][1]-box[0][1]+2*max_dim]

            # testing if the new box is overlapping with any
            # if yes then join them and add
            new_boxes_copy = []
            for pre_box in new_boxes:
                if find_overlap(new_box,pre_box):
                    # unite them
                    top_left = min(new_box[0],pre_box[0]),min(new_box[1],pre_box[1])
                    new_box = [*top_left,
                               max((new_box[0]+new_box[2]),(pre_box[0]+pre_box[2]))-top_left[0],
                               max((new_box[1]+new_box[3]),(pre_box[1]+pre_box[3]))-top_left[1]]
                else:
                    new_boxes_copy.append(pre_box)

            new_boxes_copy.append(new_box)
            new_boxes = new_boxes_copy[:]
        # splitting the overlapping groups of boxes
        # for each group find the bounding box and redetect on them
        if type(imgfile)==str:
            imgfile = cv2.imread(imgfile)
        for box in new_boxes:
            x,y,w_,h_ = tuple(box)
            cropped_img = imgfile[y:y+h_,x:x+w_]
            new_detections, _ = self.detect(cropped_img)
            p0 = (x,y)
            detections.extend(transform_detection(p0,new_detections,self.detect_thresh))
        return results_+detections,(w,h)
        # calculate the final detection and return it


if __name__ == '__main__':

    detector  = YoloDetector(os.path.join(configs.cwd,'model','yolov4-obj.cfg'),
                        os.path.join(configs.cwd,'model','Yolov4_epoch300.pth'),use_cuda=False)
    r = detector.detect(os.path.join(configs.cwd,'model','00120.jpg'))

    print(r)
    print(detector.m)
