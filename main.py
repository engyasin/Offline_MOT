import argparse
import numpy as np
import cv2

from config import config
from fix_view import fix_view
from tracking import track_objs, check_tracking, track_new
from background_substraction import bg_substract
from Detection.detection import detect
from utils import save_tracks

from objects_classes import traffic_entity

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
#ap.add_argument("-t", "--tracker", type=str, default="kcf",
#	help="OpenCV object tracker type")

args = vars(ap.parse_args())

def FirstFrame(frame):
    pass


def main():

    # read video file
    v_obj = cv2.VideoCapture(args["video"])
    ret, frame = v_obj.read()
    frame_id = 0

    # good tracks
    saved_tracks = []

    # run first frame logic
    objects = FirstFrame(frame)

    # for every frame and object in the list:
    previous_frame = frame.copy()
    ret, frame = v_obj.read()
    while frame is not None:
        frame_id += 1

        # stabilize frame by frame
        frame = fix_view(frame)

        # track
        objects = track_objs(frame,objects)

        # check tracking with background substraction
        foreground, bg = bg_substract(frame,bg)
        all_ok, objects = check_tracking(objects,foreground)

        # track everything if new objects are added by bg_substract
        if not all_ok:
            objects = track_new(objects)

        # update objects
        for obj in objects:
            obj.update()

            # deal with the newly not detected with spical logic

            # filter bad objects

        # detect every N frame,
        if (frame_id%config.N)==0:
            detections = detect(frame)

            # filter bad objects after detection
            new_objects = []
            for obj in objects:
                # object deleted if not ok here:
                ok = obj.filter_by_detections(detections)
                if ok:
                    new_objects.append(obj)
                elif obj.good_enough():
                    saved_tracks.append(obj)

            objects = new_objects

        previous_frame = frame.copy()
        ret, frame = v_obj.read()

    # save the most good tracks
    detections = detect(previous_frame)
    for obj in objects:
        ok = obj.filter_by_detections(detections)
        if ok or obj.good_enough():
            saved_tracks.append(obj)

    save_tracks(saved_tracks,args['video'])


if __name__=='__main__':
    main()
