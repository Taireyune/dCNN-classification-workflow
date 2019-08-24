import numpy as np
from tensorflow.keras.utils import Sequence
import os, cv2, random

class DataGenerator(Sequence):
    def __init__(self, InputProcess, Augmentation, batch_size, epoch_step):
        'Initialize Generator Settings'
        self.batch_size = batch_size
        self.epoch_step = epoch_step
        
        'Initialize InputProcess settings'
        self.image_size = InputProcess.image_size
        self.channels = InputProcess.channels
        self.data_dir = InputProcess.data_dir
        self.Categories = InputProcess.Categories   
        self.TrainingImages = InputProcess.CollectTrainingImages()       
        
        'Initialize Augmentation settings'
        self.Aug = Augmentation
        
        'start'
        self.on_epoch_end()
  
    def on_epoch_end(self):          
        random.shuffle(self.TrainingImages)
        
    def __len__(self):
        return int((self.epoch_step / self.batch_size) + 0.5)

    def __getitem__(self, index):
        # Generate data
        #print("__getitem__ index:", index)
        X, y = self.GetBatch(index)
        return X, y
        
    def GetBatch(self, start):
        X = np.empty((
                self.batch_size, self.image_size, 
                self.image_size, self.channels
                      ))

        y = np.empty((self.batch_size))
        for i in range(self.batch_size): 
            img_array, label = self.TrainingImages[start * self.batch_size + i]
            X[i,] = np.array(self.Aug.AugmentColoredImages(img_array))
            y[i] = label
        
        #print("   ", self.Identifier, self.currentFile, self.currentImage)
        #return X/255.0 - 0.5, y     #normalize date
        return X, y

    