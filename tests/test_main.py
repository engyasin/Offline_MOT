import unittest
import numpy as np

import cv2

from offlinemot.main import bgObjs_to_objs, FirstFrame, detections_to_objects
from offlinemot.objects_classes import TrafficObj
from offlinemot.config import config

from offlinemot.detection import YoloDetector

from offlinemot.background_substraction import BG_substractor

from tests.example_data import *
import os


class Test_main(unittest.TestCase):

    #example_obj = BG_substractor(bg=np.zeros((3,700,700)))

    
    def test_bgObjs_to_objs(self):
        bg_=np.random.rand(2000,2000,3)*200
        example_obj = BG_substractor(bg=bg_)

        bg_[100:990,100:1990,:] = 255
        fg = example_obj.bg_substract(frame=bg_)

        result = example_obj.get_big_objects(fg,bg_)

        output = bgObjs_to_objs(result[1],bg_,frame_id=1)
        self.assertEqual(len(output),1)
        # found via bg substract
        self.assertEqual(output[0].trust_level[-1][2],1)

    def test_FirstFrame(self):
        img = os.path.join('model','00120.jpg')
        output, detector = FirstFrame(cv2.imread(img))

        self.assertEqual(output[0].trust_level[-1][0],1)

    def test_detections_to_objects(self):
        frame = np.zeros((500,500,3))
        objs = detections_to_objects(detections,frame,last_track_id=0)

        N = sum([(det[2]>config.detect_thresh) for det in detections])
        self.assertEqual(len(objs),N)

        for obj in objs:
            self.assertEqual(obj.box,detect_boxes[obj.track_id])



