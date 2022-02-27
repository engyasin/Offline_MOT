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

If you have any questions or comments, or if you find any bugs, please open an issue in this project. 

Please feel free to fork the project, and create a pull request, if you have any improvements or bug fixes. 

Testing and code quality
------------------------

The code is fully documented and has good testing. In order to run the 
testing scripts, you can refer to **readme** for that. 


<!---
Style Guide
-->