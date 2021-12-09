---
title: 'OfflineMOT: A Python Package for multiple objects detection and tracking from bird view stationary drone videos'
tags:
  - Python
  - Multiple objects tracking
  - Traffic trajectories
  - Objects detection
  - Drone video analysis
authors:
  - name: Yasin Maan Yousif
    orcid: 0000-0002-5282-7259
    affiliation: 1
affiliations:
 - name: Technische Universität Clausthal, Clausthal-Zellerfeld, Germany.
   index: 1
date: 28 November 2021
bibliography: paper.bib
---

# Summary

The topic of multi objects tracking (MOT) is still considered an open research area [@MOTChallenge2015].
Among the many available methods for this problem it is worth mentioning *Deep Sort* [@wojke2017simple], where a detection and tracking steps is done in real time and for different types of scenes and areas (for example on pedestrians movement or for cars tracking). 

The importance of the problem comes from the many applications which includes, self-driving cars software, traffic analysis, animals’ movement analysis, or general surveillance applications.  
Unfortunately, due to the variety of scenes and contexts and due to the time constrains that are needed for some applications (for example, self-driving cars), there is no one general solution that is able to work reasonably for all the cases.

 `OfflineMOT` ,the package introduced here ,tries to provide a solution to a more restricted problem, namely bird’s eye stationary videos without real-time condition. It applies mainly three techniques for detection and tracking on three different priority levels.

- First level is **background subtraction**. It is a fast method to find what pixels have changed in the image (the forground), and this is plausable because of the stationary condition. Otherwise, if the drone is moving freely, then the background will be less learnable. 
Another problem here is the subtle movement of the drone due to wind and control noise. To solve this, a program for fixing the view is implemented. It depends on matching the background features with every next frame, and then transforming the frame if big movement is detected.

- Second level is **multi-objects tracking** methods such as *kernelized correlation filters* (KCF) [@henriques2014high]. This method takes the output of the detection with the next frame and it finds where the objects will be. It can also return the failure or success states of tracking.

- Third level is the normal **deep learning-based detection and classification** of the objects. Here *Yolo-v4* [@yolov4] is used as a model structure of which training is done separately. The code used to implement, load and train Yolo structure is taken from this project [@Tianxiaomo] 

Finally, all these parts are implemented separately in the code, making it easy to enable or disable some or parts of them, with many tunable parameters. This is done in purpose in order to facilitate the processing on any new video with different settings by changing only a few parameters.

The following pesudo code \autoref{fig:workflow} is summarizing the main workflow implemented in this project.

![The general workflow of the method.\label{fig:workflow}](workflow.PNG)

# Statement of need

The specific case for extracting trajectories of traffic videos (for pedestrians, cyclists and vehicles) lacks targeted  open source solutions in the literature. 
Therefore, the development of this package is directed towards helping researchers in the field of traffic analysis or any other fields where trajectories of moving objects are needed.  

This project was used to extract the trajectories from a dataset of the cyclists’ behavior in TU Clausthal successfully. The dataset itself will be published later publicly.

# Example Usage

This package can be installed simply by cloning the GitHub repository.
Additionally, a few requirements should be installed at the start. This can be done by running the following command inside the main directory:

```
$ pip install -r requirements.txt
```
The main libraries which are used includes, *OpenCV* [@opencv_library], *Numpy* [@harris2020array], *scikit-image* [@van2014scikit] , and *pytorch* [@NEURIPS2019_9015].

The main functionality of the package can be tested using any drone video or the included demo video by running the following command inside the main directory:

```
$ python main.py -v docs\sample.mp4
```

The `-v` flag is here to set the directory of the input video. The demo video of the example above is taken from [@yang2019top]. 

Several tests with the values of the parameters in `config.py` maybe needed in order to get the best results. 

A window of the real tracking status will be shown. This frame by frame result is useful for debugging. However this is not the final results of tracking, because several post processing operations will be done after finishing processing the video. 
Finally, a text file containing the results by the same name as the video will appear in the `outputs` folder. 

For example, for the previous command the following content is shown in the first line inside `sample.txt`:

` 39 || [3748, 964, 169, 73] || 2 || 5 || -137`

This line means that in frame number 39, there is a box of dimensions [169,73] with top-left point whose coordinates are [3748,964]. It is classified as 2 (cyclist), numbered with id 5 and his orientation angel is -137 degree.

In order to view the final result, the same processing command can be run but with a change of the name of the script as follows:

```
$ python show_results.py -v docs\sample.mp4
``` 

This will show the final result overlaid on the original video with customized size. 

Further documentations and information about the running are available in the `docs` folder in the format of Jupyter notebooks.

# Scope

The scope of the problems that can be handeled by this package is defined by the following conditions:

1.	*is the video Stationary?*
2.	*is the real time performance required?*
3.	*is the view of the video a bird’s eye view?*
4.	*is there a pretrained detection model for the objects of interest?*

Regarding the last point, the provided model with the package is trained on random images of cyclists, cars and pedestrians from bird’s eye view. This could be enough if the problem is the same, i.e. tracking traffic entities. Otherwise, this model could be a good starting point to train for other kinds of objects if these objects are similar and Yolo v4 is used as a model structure.

## Failure Cases

If the video is too noisy, has low resoluation, or the training dataset detection is very different from the video background and objects, then errors in tracking can happen.

As an example, the sample video has some problems with one moving object, becuase of the different background and the new scene of the video. This can be avoided by retraining the detection part on similar examples. Additionally, a through tunning step for the parameters in the `config` file should be done to eliminate the possible errors in the result. 

# References