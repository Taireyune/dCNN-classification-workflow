#%%
from preprocessing.main import data_preparation
from utils import constants as C
prep = data_preparation()

read_dir = '/media/taireyune/RAIDext4/MML1909/original_images'
error_files = prep.organize_data(read_dir)
prep.split_data(validation_volume = 1000)
# X, y = prep.in_mem_validation()

#%% debug augment images
from preprocessing.main import data_preparation
from preprocessing.image_augmentation import image_augmentation
from utils import constants as C
import cv2
import numpy as np

aug = image_augmentation(crop = True, brightness = True, gray = True, 
                         transformation = True, blurring = True, spots = True)

img_dir = '16.jpg'
img_array = cv2.imread(img_dir)

#%%
output_array = aug.augment_colored_images(img_array)

cv2.imshow('img', output_array.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows() 

#%%
from training.model_trainer import model_trainer
name = "CatDog-ColorGenData-xcept-imagenet-frozen"
train = model_trainer(name)
train.setup_model(learning_rate = 0.001)
train.run_model(20, period = 1)

#%%
from training.model_trainer import model_trainer
name = "CatDog-ColorGenData-xcept-imagenet-frozen"

load_file = 'CatDog-ColorGenData-xcept-imagenet-frozen1568147121-0003.hdf5'

train = model_trainer(name)
train.setup_model(learning_rate=0.00001)
loss, acc = train.reload_model(load_file)

train.run_model(20, period = 1)
#train.save_model(load_file)

#%%
from training.model_trainer import model_trainer

name = "CatDog-ColorGenData-xcept-imagenet-frozen"
load_file = 'CatDog-ColorGenData-xcept-imagenet-frozen1568148828-0011.hdf5'

train = model_trainer(name)
train.setup_model()
train.save_model(load_file)

#%%
from utils.predictions import predictions
from preprocessing.main import data_preparation
prep = data_preparation()

model_dir = 'CatDog-ColorGenData-xcept-imagenet-frozen-0.979.hdf5'
P = predictions(model_dir, prep.training_list)

record = P.run_prediction(threshold = 0.9)

