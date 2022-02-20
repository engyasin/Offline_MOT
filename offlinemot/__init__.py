
"""
The main package directory
"""

import pathlib, sys,os

sys.path.append(str(pathlib.Path(__file__).parent))

import utils_ 
utils_.load_model()



import background_subtraction 
import config 
import detection 
import fix_view 

import objects_classes 

import post_process 

import show_results
import main 



