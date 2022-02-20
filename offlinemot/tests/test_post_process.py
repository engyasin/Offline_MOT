import unittest
import numpy as np

from offlinemot.post_process import *
from offlinemot.objects_classes import TrafficObj

from tests.example_data import *

class Test_post_process(unittest.TestCase):

    def test_find_cntrs(self):
        
        self.assertEqual(find_cntrs(boxes),float_centers)

    def test_tracks_angels(self):
        
        self.assertEqual(tracks_angels(track),angels)
    
    def test_repair_traj(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        example_obj.centers = track
        example_obj.trust_level = trust_level
        new_track = repair_traj(example_obj)
        print(new_track)
        j = 0
        for i,condition in enumerate(trust_level):
            if any(condition):
                self.assertEqual(track[j],new_track[i])
                j += 1
            else:
                self.assertLessEqual(new_track[i-1][0],new_track[i][0])
                self.assertLessEqual(new_track[i-1][1],new_track[i][1])

