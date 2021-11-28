import unittest
import numpy as np

import cv2

from offlinemot.fix_view import FixView

from tests.example_data import *



class Test_fix_view(unittest.TestCase):

    def test_fix_view(self):
        bg_=np.uint8(np.random.rand(700,700,3)*200)
        example_obj = FixView(bg_rgb=bg_)
        # make no movement
        new_frame = example_obj.fix_view(frame=bg_)
        np.testing.assert_equal(bg_,new_frame)

    def test_get_good_matches(self):
        bg_=np.uint8(np.random.rand(700,700,3)*200)
        example_obj = FixView(bg_rgb=bg_)
        
        good_m = example_obj.get_good_matches(matches)
        self.assertEqual([x.distance for x in good_m],[1,2])

    def test_set_new_bg(self):
        bg_=np.uint8(np.random.rand(700,700,3)*200)
        example_obj = FixView(bg_rgb=bg_)
        old_kps = [kp.pt for kp in example_obj.kps_bg]

        bg_=np.uint8(np.random.rand(700,700,3)*200)
        example_obj.set_new_bg(bg_rgb=bg_)
        new_kps = [kp.pt for kp in example_obj.kps_bg]
        
        self.assertNotEqual(old_kps,new_kps)





