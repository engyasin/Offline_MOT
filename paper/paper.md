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

The topic of multi-objects tracking is considered an open research topic till now [@MOTChallenge2015].
 
Among the many methods available for this problem it is worth mentioning *Deep Sort* [@wojke2017simple], where a detection and tracking should be done in real time and for varying types of scenes and areas (for example on pedestrians movement). 

The importance of the problem comes from the many applications that can benefit from addressing it. For example, self-driving cars software, traffic analysis, animals’ movement analysis, or general surveillance applications.  
Unfortunately, due to the variety of scenes and contexts and the time constrains that are needed for the applications (for example, real-time performance condition), there is no one general solution that is able to work reasonably for all the cases.

Therefore, the work here tries to provide a solution to a more restricted problem, namely bird’s eye stationary view with no real-time performance. `OfflineMOT` is a package that does mainly apply three techniques for detection and tracking, on three different priority levels.

First level is background subtraction. It is a fast method to find what pixels have changed in the image, and this is plausable because of the stationary condition. Otherwise, if the drone is moving freely, then the background will be much harder to learn and this cannot be applied. 
One problem to mention here is the subtle movement of the drone due to wind and noise, etc. To solve this, a program to fix the view is introduced. Its idea depends on matching the first background features with every next frame, and then transforming the frame if big movement is detected.

Second level is multi-objects tracking methods such as *kernelized correlation filters* (KCF) [@henriques2014high]. This method takes the output of the detection and the next frame to find where the objects will be. It can also return the failure or success states in tracking.

Third level is the normal deep learning-based detection and classification of the objects. Here *Yolo-v4* [@yolov4] is used as a model structure of which training is done separately and not covered in the code. Additionally, the used code to implements Yolo structure and to load it is taken from this project [@Tianxiaomo] 

Finally, all these parts are implemented separately in the code, making it easy to use or disable some or parts of them, with many tunable parameters. This is done in purpose in order to facilitate the running of the code on any new video with different settings by only changing a few parameters.


# Statement of need

The specific case for extracting trajectories of traffic videos (for pedestrians, cyclists and vehicles) lacks targeted solution and open source projects in the literature. 
Therefore, the development of this package is directed towards helping researchers in the field of traffic analysis or other fields where trajectories of moving objects are needed to get these trajectories and focus their research on the main task at hand. 

This project was used to extract the trajectories of a datasets about the cyclists’ behavior in TU Clausthal successfully. The dataset itself will be published later publicly.


# Example Usage

This package can be installed simply by cloning the GitHub repository.
Additionally, a few requirements should be installed as well. This can be done by running the following command inside the main directory:

```
$ pip install -r requirements.txt
```
The main libraries which are used includes, *OpenCV* [@opencv_library], *Numpy* [@harris2020array], *scikit-image* [@van2014scikit] , and *pytorch* [@NEURIPS2019_9015].
The main functionality of the package can be tested using any video or the demo video by running the following command inside the main directory:

```
$ python main.py -v docs\sample.mp4
```

The `-v` flag is used here to set the directory of the input video. The example above is for a demo video inside the repository. 

The results will show a window of the real tracking status which is useful for debugging. Keep in mind that this is not the final results of tracking. 
After the end of the tracking a text file with the same name as the video will appear in the `outputs` folder. 

For example, for the previous command the following content is shown in the first line inside `sample.txt`:

` 39 || [3748, 964, 169, 73] || 2 || 5 || -137`

This means that in frame number 39, there is a box of dimensions [169,73] and with top-left point [3748,964] that is classified as 2 (cyclist) and numbered with id 5 and his orientation angel is -137 degree.

In order to view the final result, the same processing command can be run but with changing the name of the script only as follows:

```
$ python show_results.py -v docs\sample.mp4
``` 

This will show the final result overlaid on the video with customized size. 

Further documentations and information about the running are available in the `docs` folder in the format of Jupyter notebooks.

# Scope

The scope of the problems that this package can deal with is defined by fulfilling the following conditions:

1.	*Stationary videos*
2.	*Offline processing*
3.	*Bird’s eye view*
4.	*Pretrained detection model availability*

Regarding the last point, the model provided with the package is trained on random images of cyclists, cars and pedestrians from bird’s eye view. This can be enough if the problem is the same, i.e. tracking traffic entities. Otherwise it could be a good starting point for training if the videos include similar objects and Yolo v4 is used as a model structure.

# References