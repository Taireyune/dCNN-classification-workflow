import cv2, os, shutil, random
import numpy as np
from tensorflow.keras.utils import Sequence

from utils import constants as C

def apply_resize(img_array, size):
    # retain image ratio, resize the larger dimension to the size specified
    height, width = img_array.shape[:2]
    if height > width:
        img_array = cv2.resize(img_array, 
                               (width * size // height, size))
    elif height < width:
        img_array = cv2.resize(img_array, 
                               (size, height * size // width))
    else:
        img_array = cv2.resize(img_array, (size, size))
    return img_array  

def pad_image(img_array): 
    # pad image such that the output is a square image
    height, width = img_array.shape[:2]
    difference = height - width
    
    if difference > 0:
        img_array = cv2.copyMakeBorder(img_array, 0, 0, 
                                       difference // 2, 
                                       difference - difference  // 2, 
                                       cv2.BORDER_CONSTANT)                
    if difference < 0:
        difference *= -1
        img_array = cv2.copyMakeBorder(img_array, 
                                       difference // 2, 
                                       difference - difference // 2, 
                                       0, 0, cv2.BORDER_CONSTANT)            
    return img_array  

def rewrite_folder(folder_dir):
    # completely delete folder_dir, create an empty folder at folder_dir
    if os.path.isdir(folder_dir):
        shutil.rmtree(folder_dir)
    elif os.path.exists(folder_dir):
        os.remove(folder_dir)      
    os.mkdir(folder_dir) 

def regular_input(image_dir):
    # read image and apply modifications that would fit model input per image
    img_array = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    img_array = apply_resize(img_array, C.work_image_size)
    return pad_image(img_array)
  
def final_preprocess(img_tensor):
    if C.gpu_preprocess:
        # only cast data type if gpu resource is used for normalization
        return img_tensor.astype(np.float32)
    else:
        # normalize and cast data type before input into model
        return img_tensor.astype(np.float32) / 255 - 0.5 # normalize


def parse_image_info(img_name):
    '''
    Takes image file names and output the one_hot encoding/label for 
    classification and the string identifier for the image    
    img_name is "{:06d}-C{}-.jpg"
    '''
    return C.one_hot[int(img_name[8]),], img_name[:6] 

class data_preparation:
    def __init__(self):    
        self.unsplit_dir = os.path.join(C.main_storage, "unsplit_data")
        self.training_dir = os.path.join(C.main_storage, "training_data")
        self.validation_dir = os.path.join(C.main_storage, "validation_data")
        
        # initialize the training and validation images
        if os.path.isdir(self.training_dir) or os.path.exists(self.training_dir):
            self.training_list = os.listdir(self.training_dir)
            for i in range(len(self.training_list)):
                self.training_list[i] = os.path.join(self.training_dir, self.training_list[i])
        
        if os.path.isdir(self.validation_dir) or os.path.exists(self.validation_dir):                
            self.validation_list = os.listdir(self.validation_dir)
            for i in range(len(self.validation_list)):
                self.validation_list[i] = os.path.join(self.validation_dir, self.validation_list[i])
                
                
    def organize_data(self, read_dir): 
        """   
        organize_data reads data from read_dir and writes the accessible images to
        the appropriate folder with new labels. Note the labels
        """
        rewrite_folder(self.unsplit_dir)
        
        # naming format for images and classifiers
        img_name = "{:06d}-C{}-.jpg"
        img_name = os.path.join(self.unsplit_dir, img_name)
        
        error_files = []
        img_counter = 0
        for category in C.categories:
            full_path = os.path.join(read_dir, category)
            label_num = C.categories.index(category)
            data_list = os.listdir(full_path)
            for file in data_list:
                file_dir = os.path.join(full_path, file)
                try:                    
                    img_array = cv2.imread(file_dir, cv2.IMREAD_COLOR)
                    img_array.shape # check shape is meant to filter out bad images
                except Exception as e:
                    print(file_dir)
                    print(e)
                    error_files.append(file_dir)
                    continue   
                img_counter += 1
                cv2.imwrite(img_name.format(img_counter, label_num), img_array)
        print("number of error files:", len(error_files))
        return error_files
     
    def split_data(self, validation_volume = 1000):
        '''
        create a training and validation folder
        split the data into the two folders according the the validation volume
        specified
        '''
        rewrite_folder(self.training_dir)
        rewrite_folder(self.validation_dir)
        
        img_list = os.listdir(self.unsplit_dir)
        random.shuffle(img_list)
        volume_count = 0
        for img_name in img_list:
            from_dir = os.path.join(self.unsplit_dir, img_name)
            if volume_count < validation_volume:
                to_dir = os.path.join(self.validation_dir, img_name)
                shutil.copyfile(from_dir, to_dir)
                volume_count += 1
            else:
                to_dir = os.path.join(self.training_dir, img_name)
                shutil.copyfile(from_dir, to_dir)
                volume_count += 1
        print("total read images:", volume_count)
        
        self.training_list = os.listdir(self.training_dir)
        for i in range(len(self.training_list)):
            self.training_list[i] = os.path.join(self.training_dir, self.training_list[i])   
            
        self.validation_list = os.listdir(self.validation_dir)
        for i in range(len(self.validation_list)):
            self.validation_list[i] = os.path.join(self.validation_dir, self.validation_list[i])                
    
    def in_mem_validation(self):
        # output image features and labels for validation in memory
        total = len(self.validation_list)        
        X = np.empty((total, C.work_image_size, C.work_image_size, C.channels))
        y = np.empty((total, len(C.categories)), dtype = np.float32)
        
        for i, file in enumerate(self.validation_list):
            y[i,], _ = parse_image_info(os.path.basename(file))             
            X[i,] = regular_input(file)
        
        print("number of validation images:", total)
        return final_preprocess(X), y
                      
class data_generator(Sequence):
    '''
    Used in conjuction with keras fit_generator to feed images and labels into
    the model in training
    '''
    def __init__(self, image_list, augmentation, batch_size, epoch_step):
        # generator settings
        self.batch_size = batch_size
        self.epoch_step = epoch_step # Note epoch step is small enough for memory
        
        # initialize image directory and augmentation
        self.image_list = image_list
        self.aug = augmentation
        
        # start        
        self.on_epoch_end()
  
    def on_epoch_end(self):  
        '''
        on_epoch_end stores a set of images and labels in training_images 
        for the model at the beginning and after each epoch.
        '''
        random.shuffle(self.image_list)
        print("compiling epoch training images")
        # get new set of training images. 
        self.training_images = []
        for i in range(self.epoch_step): 
            label, _ = parse_image_info(os.path.basename(self.image_list[i]))
            img_array = cv2.imread(self.image_list[i], cv2.IMREAD_COLOR)
            self.training_images.append([img_array, label])
        
    def __len__(self):
        # find the number of index per epoch for the workers
        return int((self.epoch_step / self.batch_size) + 0.5)

    def __getitem__(self, index):
        '''
        Each worker randomly obtain an index to read the training_images
        Image is modified using augment_colored_images
        Output a batch of data for training using the index
        '''
        X = np.empty((self.batch_size, C.work_image_size, 
                      C.work_image_size, C.channels),
                     dtype = np.float32)
        
        y = np.empty((self.batch_size, len(C.categories)), dtype = np.float32)
        for i in range(self.batch_size): 
            img_array, y[i] = self.training_images[index * self.batch_size + i]
            # augment_colored_images yields float32 tensors
            X[i,] = np.array(self.aug.augment_colored_images(img_array))        
    
        return final_preprocess(X), y    # remember to change normalization settings



            
            
            
            
            
        
        
        
        

   

