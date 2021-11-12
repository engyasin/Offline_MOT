from tool import darknet2pytorch
import torch


from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

# load weights from darknet format
#model = darknet2pytorch.Darknet('yolov4-obj.cfg', inference=True)
#model.load_weights('yolov4-obj_last.weights')

# save weights to pytorch format
#torch.save(model.state_dict(), 'yolov4-pytorch.pth')

# reload weights from pytorch format
model_pt = darknet2pytorch.Darknet('../yolov4-obj.cfg', inference=True)
model_pt.load_state_dict(torch.load('../yolov4-pytorch.pth'))

use_cuda = True

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/x.names'
    #class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    plot_boxes_cv2(img, boxes[0], savename='predictions.jpg')#, class_names=class_names)

def detect(frame):

    return False


if __name__ == '__main__':
    detect_cv2('yolov4-obj.cfg','yolov4-obj_last.weights','00120.jpg')