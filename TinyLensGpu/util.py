import numpy as np
import os

def auto_mkdir_path(path_dir):
    if not os.path.exists(path_dir):
        abs_path = os.path.abspath(path_dir) 
        os.makedirs(path_dir)