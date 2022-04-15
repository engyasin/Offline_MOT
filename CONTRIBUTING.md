Contributing to Offline Multi Objects Detection and Tracking
-----------------

The OfflineMOT project is dedicated to solve the problem of 
trajectories extraction out of drone stationary videos. Of course
there are many things to be done regarding enhancing accuracy or even 
speed (even though this is not a priority). It is a research effort
aimed to help other researchers who need to collect and analysis movement data 
in multiple fields.

Current status
----------------

The software can now detect and track with occasional mistakes, mainly due
to imperfection on the detection network training model. Among some ideas
for future work, one can list:

* Better smoothing and post processing methods (including extrapolation and interpolation)
* Matching detections with object regarding other metrics (than only distance)
* Interface with other object detection network model, such Mask-RCNN

How to contribute
-----------------

Roughly, the project has 3 modules:

* Background subtraction
* Objects tracking
* Objects detection

Additionally, there's other scripts for:

* Fixing the view
* The complete detection and tracking algorithm

For addressing a bug or working in a new feature in any of these modules, please:

- Fork the project
- Work on a new branch with meaningful name
- Pull request your branch to the project with full description of the changes.


Alternatively, you can contribute by creating an issue for a problem when running the program. If your issue is about the accuracy of the results (like not detecting or failing to track some objects), please tag the issue with **logic error**. Please also attach some images or gif files depicting how the error happened in running and post-running time of the video.

Testing and code quality
------------------------

The code is fully documented and has good testing. In order to run the 
testing scripts, you can refer to **readme** for that. 

