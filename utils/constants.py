import numpy as np
import os

# directories
main_storage = '/media/taireyune/RAIDext4/MML1909/workspace_storage'
check_point_path = os.path.join(main_storage, "check_points") 

# data info
categories = ["Dog", "Cat"]
one_hot = np.identity(len(categories), dtype = np.float32)

# image settings
model_image_size = 299 
work_image_size = 128
channels = 3

# computation settings
gpu_preprocess = True