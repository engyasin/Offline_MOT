Installing
==========


The package can be installed on python 3.x simply using the ``pip`` command:

.. code-block:: python

    pip install offlinemot


Getting Started
----------------

After installing the library, and in order to test the example provided with it, the following lines can be used in as python commands:

.. code-block:: python

    In [1]: import offlinemot

    In [2]: offlinemot.main.main() # no input to run the example video


In the first time this is ran, the example network model will be downloaded (around 250MB). And a window for the example video with the tracked objects will be shown.

The tracked objects will be surrounded with boxes in 5 different colors. Each color has a spicific meaning:

.. raw:: html

    <ul>
    <li> <span style="color:green">Green</span>: Pedestrian is detected. </li>
    <li> <span style="color:blue">Blue</span>: Cyclist is detected.</li>
    <li> <span style="color:black">Black</span>: Car is detected.</li>
    <li> <span style="color:red">Red</span>: The tracked object has failed the tracking step for the current frame</li>
    <li> <span style="color:gray">White</span>: The object is moving but still not classified to a class.</li>
    </ul>

Of course, for a different case, the colors can be changed from the ``config.py`` file_. This also depends on the number of classes to predict.

To control these parameters and many others, the commands to be run are:

.. code-block:: python

    offlinemot.main.set_params()


This will open a new text editor with the ``config.py`` file containing all of the parameters.

.. note:: 
    It's highly recommended to set all the parameters when running on a new video. 
    A detailed description for their meaning is available in the ``config`` file_ . 
    Additionally, a complete example for parameters tuning is available here link_ 

.. _link: tutorials/A_Working_Example.ipynb
.. _file: generated/offlinemot.config.config.html

Running
----------

To run it on some video, the command is:

.. code-block:: python

    offlinemot.main.main('path_to_video') 
    #[directory of the videos, leave empty to run the example video]

To show the result on the same video after post processing, use the command:

.. code-block:: python

    offlinemot.show_results.main('path_to_same_video')
    #[directory of the videos, leave empty to run the example video]


Finally, to change the yolo network used in the package, the complete directory to 3 files need to be assigned inside the ``config.py``:

- *.pth* for the model weights
- *.cfg* for the Yolo configuration.
- *.names* for a simple text file containing the names of the classes.
