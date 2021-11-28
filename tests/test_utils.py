import unittest
import numpy as np

from offlinemot.utils import *

from tests.example_data import *

class Test_utils(unittest.TestCase):

    def test_resize(self):
        img = np.random.rand(50,50,3)
        result = resize(img,scale=0.5)
        self.assertEqual(result.shape,(25,25,3))

    def test_check_box(self):

        self.assertEqual(check_box(boxes[0], img_whs[0]),False)
        self.assertEqual(check_box(boxes[1], img_whs[1]),True)
        self.assertEqual(check_box(boxes[2], img_whs[2]),False)
        self.assertEqual(check_box(boxes[3], img_whs[3]),False)

    def test_find_overlap(self):

        self.assertEqual(find_overlap([0,0,10,5],[0,0,5,10]),25)
        self.assertEqual(find_overlap([0,0,10,5],[15,15,5,10]),0)


    def test_transform_detection(self):


        self.assertEqual(transform_detection(points[0],detections),results_0)


        self.assertEqual(transform_detection(points[1],detections),results_1)


        self.assertEqual(transform_detection(points[2],detections),results_2)
