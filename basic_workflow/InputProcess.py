import cv2, os, shutil
import _pickle as pickle
import numpy as np

class InputProcess:
    def __init__(self, main_storage, image_size, Categories, is_colored = True):
        self.main_storage = main_storage # main data storage directory 
        self.Categories = Categories
        self.image_size = image_size
        self.is_colored = is_colored
        
        if self.is_colored:
            self.imread_setting = cv2.IMREAD_COLOR
            self.channels = 3
        else:
            self.imread_setting = cv2.IMREAD_GRAYSCALE
    
        self.data_dir = os.path.join(self.main_storage, "training_data")
        self.save_pickle_dir = os.path.join(self.main_storage, "training_data.pickle")
        
    """   
    ObtainCleanData makes a copy of the whole data set to the appropriate folder.
    Screen the folder for bad data files and remove the hits.
    """
    def ObtainCleanData(self, read_dir):       
        if os.path.isdir(self.data_dir):
            shutil.rmtree(self.data_dir)
        elif os.path.exists(self.data_dir):
            os.remove(self.data_dir)      
        shutil.copytree(read_dir, self.data_dir)
        
        error_files = []
        for category in self.Categories:
            path = os.path.join(self.data_dir, category)
            data_list = os.listdir(path)
            for file in data_list:
                file_dir = os.path.join(path, file)
                try:
                    img_array = self.GetTestImage(file_dir)
                except Exception as e:
                    print(file_dir)
                    print(e)
                    error_files.append(file_dir)
                    os.remove(file_dir)
                    continue
                
        print(len(error_files))
        return error_files
    
    '''for classification data folders only'''
    def CompileTrainingImages(self):
        TrainingImages = []
        for category in self.Categories:
            category_dir = os.path.join(self.data_dir, category)
            class_num = self.Categories.index(category)
            for file_name in os.listdir(category_dir):
                file_dir = os.path.join(category_dir, file_name)
                img_array = cv2.imread(file_dir, self.imread_setting)
                img_array = self.ApplyResize(img_array)
                TrainingImages.append([img_array, class_num])
        
        with open(self.save_pickle_dir, "wb") as save_file:
            pickle.dump(TrainingImages, save_file)           
        print("saved training data pickle file:", self.save_pickle_dir)
        return TrainingImages
             
    def CollectTrainingImages(self):
        if not os.path.exists(self.save_pickle_dir):
            print("No validation file found")
            return False

        with open(self.save_pickle_dir, "rb") as open_file:
            return pickle.load(open_file)
        
    def ApplyResize(self, img_array):
        height, width = img_array.shape[:2]
        if height > width:
            new_width = int(width * self.image_size / height)
            new_height = self.image_size
            img_array = cv2.resize(img_array, (new_width, new_height))
        elif height < width:
            new_height = int(height * self.image_size / width)
            new_width = self.image_size
            img_array = cv2.resize(img_array, (new_width, new_height))
        else:
            new_height = self.image_size
            new_width = self.image_size
            img_array = cv2.resize(img_array, (new_width, new_height))
        return img_array               
        
    def GetTestImage(self, image_dir):
        img_array = cv2.imread(image_dir, self.imread_setting)
        img_array = cv2.resize(img_array, (self.image_size, self.image_size))
        img_array = self.PadImage(img_array)
        return img_array
 
    def PadImage(self, img_array):  
        if self.is_colored:
            height, width, a = img_array.shape
        else:
            height, width = img_array.shape
            
        difference = height - width
        ratio = height/width
        
        if difference > 0:
            pad = int(difference/2 + 0.5)
            if ratio > 4/3:
                img_array = cv2.copyMakeBorder(img_array, 0, 0, pad, pad, cv2.BORDER_CONSTANT)
            else:
                img_array = cv2.copyMakeBorder(img_array, 0, 0, pad, pad, cv2.BORDER_CONSTANT)          
            
        if difference < 0:    
            pad = int(-difference/2 + 0.5)
            if ratio < 3/4:
                img_array = cv2.copyMakeBorder(img_array, pad, pad, 0, 0, cv2.BORDER_CONSTANT)
            else:
                img_array = cv2.copyMakeBorder(img_array, pad, pad, 0, 0, cv2.BORDER_CONSTANT)
            
        return img_array   

    
