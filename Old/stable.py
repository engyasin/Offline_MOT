import os
import cv2
from vidstab import VidStab
import imutils


# Initialize object tracker, stabilizer, and video reader
stabilizer = VidStab()
vidcap = cv2.VideoCapture('../../DJI_0148.mp4')

# Initialize bounding box for drawing rectangle around tracked object
object_bounding_box = None

while True:
    grabbed_frame, frame = vidcap.read()
    frame = imutils.resize(frame, width=600)

    # Pass frame to stabilizer even if frame is None
    stabilized_frame = stabilizer.stabilize_frame(input_frame=frame, border_size=20,smoothing_window=10)
    # If stabilized_frame is None then there are no frames left to process
    if stabilized_frame is None:
        break

    # Display stabilized output
    cv2.imshow('Frame', stabilized_frame)

    key = cv2.waitKey(5)

    # Select ROI for tracking and begin object tracking
    # Non-zero frame indicates stabilization process is warmed up

    if key == 27:
        break

vidcap.release()
cv2.destroyAllWindows()