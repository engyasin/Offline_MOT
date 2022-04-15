
import unittest
import numpy as np

import cv2
import os
from offlinemot.detection import YoloDetector

from tests.example_data import *
from config import configs



class Test_detection(unittest.TestCase):

    def test_detect(self):
        
        detector  = YoloDetector(config=configs())
        r = detector.detect(os.path.join(configs.cwd,'model','00120.jpg'))

        # it is six objects
        ped,cycle = 0,0
        for obj in r[0]:
            if obj[3]==1:
                ped += 1
            elif obj[3]==2:
                cycle += 1

        self.assertEqual(len(r[0]),6)
        self.assertEqual(ped,3)
        self.assertEqual(cycle,3)

    def test_better_detection(self):
        detector  = YoloDetector(config=configs())

        r = detector.better_detection(os.path.join(configs.cwd,'model','00120.jpg'))

        # it is twelve objects (duplicate)
        self.assertEqual(len(r[0]),12)

