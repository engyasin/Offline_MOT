import unittest
import numpy as np

import cv2

from offlinemot.objects_classes import TrafficObj

from tests.example_data import *

class Test_objects_classes(unittest.TestCase):

    #example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)

    def test_find_center(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)       
        self.assertEqual(example_obj.find_center(),(550,450))

    def test_track(self):
        example_obj = TrafficObj(frame=np.ones((700,700))*255,frame_id=0,box=boxes[0],track_id=1,tracker=cv2.TrackerKCF_create)       

        frameid = 1
        example_obj.track(new_frame=np.zeros((700,700)),frame_id=frameid)
        #tracking should fail here
        state = example_obj.tracking_state[-1]
        self.assertEqual(example_obj.time_steps[-1],frameid)
        self.assertEqual(example_obj.trust_level[-1],[0,int(state),0])

    def test_draw(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1,class_id=-1) 

        new_frame = np.zeros((3,700,700))
        target_frame = new_frame.copy()

        x,y,w,h = tuple(boxes[0])
        cv2.rectangle(target_frame, (x, y), (x + w, y + h), color=(255,255,255), thickness=4)

        np.testing.assert_equal(example_obj.draw(new_frame),target_frame)

    def test_update(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        self.assertEqual(example_obj.update(),(True,False))


    def test_filter_by_detections_dist(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        self.assertEqual(example_obj.filter_by_detections_dist(detections,check=False),
                        (False,detections))


    def test_set_detection(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        example_obj.set_detection(True)
        self.assertEqual(example_obj.trust_level[-1][0],int(True))

    def test_set_bg_substract(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        example_obj.set_bg_substract(True)
        self.assertEqual(example_obj.trust_level[-1][2],int(True))

    def test_set_track_id(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  
        example_obj.set_track_id(id_=5)
        self.assertEqual(example_obj.track_id,1)

    def test_find_true_size(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1)  

        example_obj.find_true_size(new_box=boxes[1])
        self.assertEqual(example_obj.true_wh_max,(boxes[1][2:],75))


    def test_get_detection_format(self):
        example_obj = TrafficObj(frame=np.random.rand(700,700)*255,frame_id=0,box=boxes[0],track_id=1,class_id=-1)

        self.assertEqual(example_obj.get_detection_format(),[(500,400),(600,500),0.5,-1])



