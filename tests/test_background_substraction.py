import unittest
import numpy as np

import cv2

from offlinemot.background_substraction import BG_substractor

from tests.example_data import *



class Test_background_substraction(unittest.TestCase):

    #example_obj = BG_substractor(bg=np.zeros((3,700,700)))

    def test_bg_substract(self):

        bg_=np.uint8(np.random.rand(700,700,3)*200)
        example_obj = BG_substractor(bg=bg_)

        bg_[100:400,100:400,:] = 255
        fg = example_obj.bg_substract(frame=bg_)

        self.assertEqual((fg[150:350,150:350]==255).all(),True)

    
    def test_get_big_objects(self):
        bg_=np.random.rand(2000,2000,3)*200
        example_obj = BG_substractor(bg=bg_)

        bg_[100:990,100:1990,:] = 255
        fg = example_obj.bg_substract(frame=bg_)

        result = example_obj.get_big_objects(fg,bg_)

        np.testing.assert_equal(result[0],fg)
        self.assertEqual(len(result[1]),1)









