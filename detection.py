from tool import darknet2pytorch
import torch


from models import Yolov4

from tool.utils import *
from tool.torch_utils import *
#
# from tool.darknet2pytorch import Darknet
import cv2

# load weights from darknet format
#model = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
#model.load_weights('yolov4-obj_last.weights')

# save weights to pytorch format
#torch.save(model.state_dict(), 'yolov4-pytorch.pth')

# reload weights from pytorch format
#model_pt = darknet2pytorch.Darknet('../yolov4-obj.cfg', inference=True)
#model_pt.load_state_dict(torch.load('../yolov4-pytorch.pth'))


class YoloDetector:

    def __init__(self,cfgfile, weightfile, use_cuda=True):

        self.m = darknet2pytorch.Darknet(cfgfile, inference=True)

        #self.m = Yolov4(yolov4conv137weight=weightfile, n_classes=3, inference=True)

        self.m.load_state_dict(torch.load(weightfile))
        #self.m.load_weights(weightfile)
        """
        pretrained_dict = torch.load(weightfile)

        model_dict = self.m.state_dict()
        for x in model_dict.copy():
            if 'neek' in x:
                del model_dict[x]
        # 1. filter out unnecessary keys
        pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        self.m.load_state_dict(model_dict)       
        
        """

        self.use_cuda = use_cuda

        #m.print_network()
        #m.load_weights(weightfile)
        #print('Loading weights from %s... Done!' % (weightfile))

        if self.use_cuda:
            self.m.cuda()

        #num_classes = m.num_classes
        #if num_classes == 20:
        #    namesfile = 'data/voc.names'
        #elif num_classes == 80:
        #    namesfile = 'data/coco.names'
        #else:
        namesfile = 'obj.names'
        self.class_names = load_class_names(namesfile)
    
    def detect(self,imgfile):
        if type(imgfile)==str:
            img = cv2.imread(imgfile)
        else:
            img = imgfile.copy()
        #img = cv2.imread(imgfile)
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
            print('class id: ',class_id)
        #print('**********')

        return results,(w,h)
    #plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)





def detect_cv2(cfgfile, weightfile, imgfile,use_cuda=True):

    #m = Darknet(cfgfile)

    m = darknet2pytorch.Darknet(cfgfile, inference=True)
    m.load_state_dict(torch.load(weightfile))

    #m.print_network()
    #m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    #num_classes = m.num_classes
    #if num_classes == 20:
    #    namesfile = 'data/voc.names'
    #elif num_classes == 80:
    #    namesfile = 'data/coco.names'
    #else:
    namesfile = 'obj.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)

    """
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))   
    """


    # print(boxes)
    # boxes : x=(0)*w, y=(1)*h, x2, y2, prob, class(0,1,2)

    h,w = img.shape[1:]
    results = []
    for box in boxes[0]:
        p1 = box[0]*w , box[1]*h
        p2 = box[2]*w , box[3]*h
        prob = box[5]
        class_id = box[6]
        results.append((p1,p2,prob,class_id))
        #print('**********')

    return results
    #plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)





if __name__ == '__main__':

    detector  = YoloDetector('yolov4-obj.cfg','INTERRUPTED.pth',use_cuda=False)
    r = detector.detect('P1_02_03_04_00001.jpg')#'00120.jpg')

    #r = detect_cv2('yolov4-obj.cfg','yolov4_last.pth','00120.jpg',use_cuda=False)
    #print(len(r))

    print(r)
    print(detector.m)
