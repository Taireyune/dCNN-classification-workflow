import os, time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

import numpy as np
from tensorflow import GPUOptions, Session, ConfigProto
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

#from tensorflow.keras.backend import cast, set_floatx, set_epsilon, cast_to_floatx
from preprocessing.main import data_preparation, data_generator
from preprocessing.image_augmentation import image_augmentation
from model.model_arch import model_tail, resize_and_normalize
from utils import constants as C

class model_trainer:
    def __init__(self, name):        

    #    tf.config.experimental.set_per_process_memory_fraction(1)
    #    tf.config.gpu.set_per_process_memory_growth(True)
        
        GPUsetting = GPUOptions(per_process_gpu_memory_fraction = 1, allow_growth = True)
        sess = Session(config = ConfigProto(gpu_options = GPUsetting))
        
        self.name = name    
        prep = data_preparation()    
        
        # feature and labels are float32 and normalized
        self.X_test, self.y_test = prep.in_mem_validation()  
        
        batch_size = 64
        epoch_step = 5000
        self.tensor_shape = (batch_size, C.work_image_size, C.work_image_size, C.channels)
        
        aug = image_augmentation(dimension_shift = True, crop = True, brightness = True, gray = True, 
                                 transformation = True, blurring = True, spots = True)       
        self.generator = data_generator(prep.training_list, aug, batch_size, epoch_step)
        # prep.training_list contains all the images directories for training
    
    def setup_model(self, learning_rate = 0.001):
        # model structure goes here                
        input_tensor = Input(self.tensor_shape[1:])        
        # remember to turn final_preprocess back on if on-GPU normalization is turned off
        x = Lambda(resize_and_normalize, 
                   arguments = {'size': (C.model_image_size, C.model_image_size)}, 
                   name = 'resize_and_normalize')(input_tensor)
        base_model = Xception(input_tensor = x, weights = "imagenet", include_top = False, input_shape = (299, 299, 3))        
        x = base_model.output
        output_tensor = model_tail(x, len(C.categories))
#        input_tensor = Input(self.tensor_shape[1:])
#        x = model_head(input_matrix)
#        x = model_main(x)
#        output_matrix = model_tail(x, len(C.categories))
        
        # compile model        
        self.model = Model(inputs = input_tensor, outputs = output_tensor)
        
        for layer in base_model.layers:
            layer.trainable = False
            
        Opt = Adam(lr = learning_rate)     
        self.model.compile(loss = "categorical_crossentropy", optimizer = Opt, 
                           metrics = ["accuracy"])   
        # self.model.summary()
        
    def run_model(self, epochs, period = None):
        # initialize tensorboard and checkpoint
        log_name = self.name + str(int(time.time()))
        tensorboard = TensorBoard(log_dir = 'log/{}'.format(log_name))   
            
        
        check_point_name = os.path.join(C.check_point_path, log_name + "-{epoch:04d}.hdf5")
        check_point = ModelCheckpoint(check_point_name, verbose = 1, 
                                      save_weights_only = True, period = period)
        
        # must evaluate to initialize everything
        loss, acc = self.model.evaluate(self.X_test, self.y_test)     
        
        self.model.fit_generator(self.generator, validation_data = (self.X_test, self.y_test),
                                 use_multiprocessing = True, callbacks = [tensorboard, check_point],
                                 epochs = epochs, workers = 6)

    def reload_model(self, check_point_name, complete_model = False):
        self.model.load_weights(os.path.join(C.check_point_path, check_point_name))
        
        # must rerun model before restarting fit
        loss, acc = self.model.evaluate(self.X_test, self.y_test)
        return loss, acc

    def save_model(self, check_point_name):
        loss, acc = self.reload_model(os.path.join(C.check_point_path, check_point_name))
        
        save_dir = os.path.join(C.check_point_path, 
                                self.name + '-{:.4f}.hdf5'.format(acc))
        self.model.save(save_dir)
        print(save_dir)
        