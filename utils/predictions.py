import os, time, cv2, shutil, random
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

import numpy as np
from tensorflow import GPUOptions, ConfigProto, Session
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model

from model.model_arch import resize_and_normalize
from preprocessing.main import rewrite_folder, regular_input, final_preprocess, parse_image_info
from utils import constants as C

class predictions:
    def __init__(self, model_dir, img_list):
        GPUsetting = GPUOptions(per_process_gpu_memory_fraction = 1, allow_growth = True)
        sess = Session(config = ConfigProto(gpu_options = GPUsetting))
        
        self.model_dir = os.path.join(C.check_point_path, model_dir)
                
        self.batch_size = 64        
        self.img_list = img_list
        self.list_size = len(self.img_list)

        self.model = load_model(self.model_dir, custom_objects = {'resize_and_normalize': resize_and_normalize})
    
    def run_prediction(self, threshold = 0.5):
        X = np.empty((self.list_size, C.work_image_size, C.work_image_size, C.channels), dtype = np.float32)
        y =  np.empty((self.list_size, len(C.categories)), dtype = np.float32)  
        
        for i, file in enumerate(self.img_list):
            y[i,], _ = parse_image_info(os.path.basename(file))             
            X[i,] = regular_input(file)
        
        X = final_preprocess(X)

        record = self.predict(X, y, threshold = threshold)
        print("# of errors:", len(record))
        print("accuracy:", len(record)/self.list_size)
        self.save_images(record)
        return record

    def predict(self, X, y, threshold = 0.5):
        print("running inference...")
        y_predict = self.model.predict(X, batch_size=self.batch_size, 
                                       verbose=0, steps=None, callbacks=None)   
        dy = np.abs(y - y_predict)
        norm_dy = np.linalg.norm(dy, axis = 1)
        recorder = []
        for i in range(self.list_size):
            if norm_dy[i] > threshold:
                recorder += [[i, y[i,], y_predict[i,]]]
        return recorder
    
    def save_images(self, records): 
        if self.model_dir.endswith('.hdf5'):
            predictions_dir = self.model_dir.replace('.hdf5', '-failed_predictions')
        else:
            print('file name error')
            return
        
        rewrite_folder(predictions_dir)
        
        for [i, y, y_predict] in records:
            file_name, suffix = os.path.basename(self.img_list[i]).split('.')
            file_name += 'PA{:.3f}-PB{:.3f}-.' + suffix
            file_name = os.path.join(predictions_dir, file_name)
            shutil.copyfile(self.img_list[i], file_name.format(y_predict[0], y_predict[1]))
