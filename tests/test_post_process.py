import unittest
import numpy as np

from offlinemot.post_process import *

from tests.example_data import *

class Test_post_process(unittest.TestCase):

    def test_find_cntrs(self):
        
        self.assertEqual(find_cntrs(boxes),float_centers)

    def test_tracks_angels(self):
        
        self.assertEqual(tracks_angels(track),angels)


