Contributing to Offline Multi Objects Detection and Tracking
=====================

The Offline MOT project is dedicated to solve the problem of 
trajectories extraction out of drone stationary videos. Of course
there are many things to be done regarding enhancing accuracy or even 
speed (even though this is not a priority). It is a research effort
aimed to help other researchers how need to collect and analysis data 
in multiple fields.

Current status
--------------

The software can now detect and track with occasional mistakes. Mainly due
to imperfection on the detection network training model. Among some ideas
for future work, one can list:

* Better smoothing and post processing methods (including extrapolation and 
interpolation)

* matching detections with object regarding other metrics (than distance)

* Interface with other object detection network model, such Mask-RCNN

How to contribute
-----------------

New contributions are welcome!  You can as well read some
documentation material first or try browsing through the code.

Testing and code quality
------------------------

The code is fully documented and has good testing. In order to run the 
testing scripts, you can refer to **readme** for that. 

Style guide
-----------
TODO
