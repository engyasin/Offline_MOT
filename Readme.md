Multiple objects detection and tracking from bird view stationary drone videos
=========
[![GH Actions Status](https://github.com/engyasin/Offline_MOT/workflows/PyTest/badge.svg)](https://github.com/engyasin/Offline_MOT/actions?query=branch%3Amain)
[![codecov](https://codecov.io/gh/engyasin/Offline_MOT/branch/main/graph/badge.svg)](https://codecov.io/gh/engyasin/Offline_MOT/branch/main)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

`OfflineMOT` is a package for multi objects tracking from bird's eye view stationary videos. The accuracy has priority over runtime in this package, therefore it is better suited for offline processing rather than real time applications, hence the name of the package.

A pretrained Yolo network is used for detection in this package and it should be trained separately. The network included with this library is Yolo v4 in a pytorch format (trained to detect pedestrians, cyclists and cars). The loading and running of Yolo model is done with the help of scripts taken from [this project](https://github.com/Tianxiaomo/pytorch-YOLOv4) (All of them are in *offlinemot/tool* subfolder)

Example output for a sample video, taken from [**Vehicle-Crowd Interaction  (VCI) - CITR Dataset**](https://github.com/dongfang-steven-yang/vci-dataset-citr) :

![Problem loading the gif!](docs/sources/../source/_static/output.gif)


***This example shows some minor problems because the scene, the background and the drone's camera are outside the detection network training set (never seen before by the detection network).***

***However, the application of this project (including the Yolo network training) was targeted for a Cyclists dataset videos [to be cited later].***

--------------------

## Installation

The package can be installed on python 3.x simply using the `pip` command:

```
pip install offlinemot
```
--------------------
## Documentation

The documentation includes some example and guides to run this package and it is available here https://engyasin.github.io/Offline_MOT/

--------------------
## Getting Started

After installing the library, and in order to test the example provided with it, the following lines can be used in as python commands:

```python
In [1]: import offlinemot

In [2]: from offlinemot.config import configs

In [3]: cfg= configs() # if you have avaliable configuration file '.ini', you can pass it

In [4]: cfg.print_summary() # show the current values and sections

In [5]: cfg['detect_every_N'] = 3

In [6]: cfg.print_section('Detection') # show parameters of single section

In [7]: cfg['detect_thresh'] = 15

In [8]: offlinemot.main.main(config=cfg) # no input to run the example video

In [9]: cfg.write('new_config_file.ini') # to be loaded for similar videos

```

For the first time this is ran, the example network model will be downloaded (around 250MB). And a window for the example video with the tracked objects will be shown.

The tracked objects will be surrounded with boxes in 5 different colors. Each color has a spicific meaning:

- <span style="color:green">Green</span>: Pedestrian is detected.
- <span style="color:blue">Blue</span>: Cyclist is detected.
- <span style="color:black">Black</span>: Car is detected.
- <span style="color:red">Red</span>: The tracked object has failed the tracking step for the current frame
- <span style="color:white">White</span>: The object is moving but still not classified to a class.

Of course, for a different case, the colors can be changed from the `cofig.py` file. This also depends on the number of classes to predict.

to control these parameters and many others, the commands to be run are:

```python
offlinemot.main.set_params()
```

Then a new text editor with the `config.py` file containing all the parameters is shown.

Note: It's highly recommended to set all the parameters when running on a new video. A detailed description for their meaning is available in the `config` file. Additionally, a complete example for parameters tuning is available in the [documentation here](https://engyasin.github.io/Offline_MOT/html/tutorials/A_Working_Example.html)

### Running

Then to run it on some video, the command is:

```python
offlinemot.main.main('path_to_video') 
#[directory of the videos, leave empty to run the example video]
```
to show the result on the same video after post processing, use the command:

```python
offlinemot.show_results.main('path_to_same_video')
#[directory of the videos, leave empty to run the example video]
```

Finally, to change the yolo network used in the package, the complete directory to 3 files need to be assigned inside the `config.py`:

- *.pth* for the model weights
- *.cfg* for the Yolo configuration.
- *.names* for a simple text file containing the names of the classes.

---------------------
## Use cases

This project can be used for:

* Traffic trajectories extraction from videos (It is originally built to extract trajectories for a cyclist's dataset for traffic modelling research recorded in TU-Clausthal).

* Tracking other objects (like animals) from bird's eye view in an offline manner.

--------------------

## Testing

There are a number of test units for this project. If a development of the package is intended then they can be run after cloning this repo with the command:
```
$ pytest -v ./offlinemot/tests
```

For the previous command `pytest` library is needed to be installed.

--------------------

## Support

If you like to contribute to a feature of a bug fix, please take a look at the [contribution instructions](CONTRIBUTING.md) has further details.


Alternatively, you can contribute by creating an issue for a problem when running the program. If your issue is about the accuracy of the results (like not detecting or failing to track some objects), please tag the issue with **logic error**. Please also attach some images or gif files depicting how the error happened in running and post-running time of the video.

--------------------

## Citation Info
To be added

--------------------
## Stars

Please star this repository if you find it useful, or use it as part of your research.

--------------------
## License
`OfflineMOT` is free software and is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). Copyright (c) 2022, Yasin Yousif 
