from tool import darknet2pytorch
import torch

from tool.utils import *
from tool.torch_utils import *

from config import config
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

    """

    def __init__(self,cfgfile, weightfile, use_cuda=True):

        """
        Parameters
        ----------
        cfgfile : str
            The filename of the configuration file of the Yolov4 model

        weightfile : str
            The filename of the network model (Yolov4)

        use_cuda : bool
            Run on CPU if False or GPU if True (default is True)

        """

        self.m = darknet2pytorch.Darknet(cfgfile, inference=True)

        self.m.load_state_dict(torch.load(weightfile))
        #self.m.load_weights(weightfile)

        self.use_cuda = use_cuda

        if self.use_cuda:
            self.m.cuda()

        namesfile = config.classes_file_name
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
            the original image. The result list has the following shape 
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


if __name__ == '__main__':

    detector  = YoloDetector('yolov4-obj.cfg','Yolov4-epoch300.pth',use_cuda=False)
    r = detector.detect('P1_02_03_04_00001.jpg')#'00120.jpg')


    print(r)
    print(detector.m)
