Installing
==========

.. note::

    The source code contains big files for the trained Yolo network.  In case you want to run the examples with the pretrained networks, then **git lfs** (large files storage) is required to be installed first. Instructions for installing *git lfs* are found in [this website](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
    Alternativaly, the model file `Yolov4_epoch300.pth` can be downloaded from [google drive here](https://drive.google.com/file/d/1rhDaY7aVSeETP8rHgqZTewp4QkWlr3fb/view?usp=sharing). It should be placed in the */model* subfolder of the package.
    
    After cloning this project, and changing the detection model (in case you need to detect other objects than cyclists, pedestrians and cars), the requirements packages should be installed, so simply, cd to the root of this project and run:

```
pip install -r requirements.txt
```
