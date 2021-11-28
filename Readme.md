
Multiple objects detection and tracking from bird view stationary drone videos
=========


This project is used to extract trajectories as precise as possible for different entites in a mixed traffic scene, from drone videos taken while hovering in the same position (stationary).

The project needs a pretrained network for detection, which should be trained separately. The network included with this library is Yolov4 in a pytorch format. The loading and running of Yolo model is done with the help of scripts taken from [this project](https://github.com/Tianxiaomo/pytorch-YOLOv4). All of these scripts are in *tool* subfolder.

<video width="80%" height="50%" controls>
  <source src="./docs/output.mp4" type="video/mp4">
Your browser does not support the video tag.
</video>

## Features of this work:

Drone videos are subject to some shaking due to some factors like noise in the control or wind. For that a fixing step for all the transformed frames is needed. 

Also performing the detection with the model for every frame is slow and can be substitued by the usage of a background substarction method to detect all moving object because the background is the same for the whole video.

At the end and because the work is offline, additional filtering steps can be done to find: 

- The true size of each object (which should be fixed because it is taken from bird's eye view)
- The orientation of all the objects
- Solving the errors of misclassifying some objects by assigning the highst and most probable detected class to them. 
- Smooth trajectories.

All this can be done because of the specific features of the problem, namely (bird's eye view, offline and 
stationary camera)

## Getting Started

After cloning this project, and changing the detection model (in case you need to detect other objects than cyclists, pedistrains and cars), the requirerments pakages should be installed, so simply, cd to the root of this project and run:

```
pip install -r requirements.txt
```

Then to run it on some video, the command is:

```
python offline_mot\main.py -v [directroy of the videos, ex: docs\sample.mp4]
```
to show the result on the same video after post processing, just replace `main.py` by `show_results.py` in the previous command. i.e:

```
python offline_mot\show_results.py -v [directroy of the videos, ex: docs\sample.mp4]
```

There are many parameters that you may want to tune before running, they are in the `offline_mot/config.py` file. The explaination is avaliable on the same file as well as in the docs folder, namely [Working example](./docs/A_Working_Example.ipynb)


This project can be used for:

* Traffic trajectories extraction from videos (It was applied successfuly to extract trajectories from the cyclists datasets for traffic modelling research recorded in TU-Clausthal).

* Tracking other objects (like animals) from bird's eye view in an offline manner.


## Workflow

Three methods are applied for detection and tracking in this project, namely in the order of thier priority:

* Background Substraction: This method are used on every frame to detect the forground objects which contain any moving object. If these objects are already tracked then nothing happen. Otherwise, it would be added and tracked as candidate objects (white boxes)

* Tracking with a filter like KCF (Kernelized Correlation Filter), which only needs the first bounding box of the object to track. These objects will continue to be tracked as long as the tracker keep giving results successfuly. Otherwise, the object will not be updated to a new position and a detection step is performed

* Detection with a network model like Yolo: This is performed only for every *N* frame as set in the `config.py` file. If the object is already tracked then it is confirmed or set to a class type, otherwise nothing happen (only a message saying that something is detected but wasn't there previously)

All these three steps are done for every object and the result is recorded for every frame. If one object keep failing all the steps then it will be deleted after a defined number of times.

## Examples and Documentation

There are a few jupyter notebook showcases for the different tracking and detection handling programs, and an additional working example for how to run and set the different parameters, Namely:

1. Background substraction example [here](./docs/Background_Subtraction_Example)

2. Tracking example [here](./docs/Tracking_Example)

3. Fixing the view example [here](./docs/Fixing_the_view)

4. A general working example [here](./docs/A_Working_Example.ipynb)


--------------------

## Testing

There are a number of test units for this project. To run the tests use the command:
```
$ python -m pytest tests
```

For the previous command you need `pytest` library installed.

--------------------

## Support

If you have any questions or comments, or if you find any bugs, please open an issue in this project. Please feel free
to fork the project, and create a pull request, if you have any improvements or bug fixes. We welcome all feedback and
contributions.

--------------------
## Citation Info
To be added

## Stars

Please star this repository if you find it useful, or use it as part of your research.

## License

[MIT License](https://choosealicense.com/licenses/mit/)