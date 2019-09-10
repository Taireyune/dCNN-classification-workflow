import numpy as np
import cv2, random
from utils import constants as C

class image_augmentation:
    def __init__(self, dimension_shift = True, crop = True, brightness = True, gray = True, 
                 transformation = True, blurring = True, spots = True):       
        # initialize width or height resize
        self.dimension_shift = dimension_shift
        self.max_shrink_ratio = 2 # 
        
        # initialize crop.
        self.crop = crop
        self.ratio_threshold = 3/2
        
        # initialize gray.
        self.gray = gray
        
        # initialize padding, rotating and flipping
        self.transformation = transformation
        if transformation:
            self.pad = [-1, 0, 1]
        else:
            self.pad = [0]
        self.flip = [0,1]
        self.rotation_matrix = []        
        for i in range(4):
            matrix = cv2.getRotationMatrix2D((C.work_image_size/2, C.work_image_size/2), 
                                             90 * i, 1)        
            self.rotation_matrix.append(matrix)

        #create a list of brightness tables
        self.brightness = brightness    
        self.brightness_tables = []
        for brightness_value in np.arange(0.5, 1.5, 0.1).tolist():
            table  = self.LUT_table(brightness_value)
            self.brightness_tables.append(table)
      
        # setup channel for gaussian noise
        self.blurring = blurring
        self.gauss_variance = [0]*10 + [i for i in range(5,11)]
        
        # salt and peppering
        self.spots = spots
        self.spot_probability = [0]*30 + np.arange(0.015, 0.031, 0.001).tolist() 
        
    "Execute this to start the run"
    def augment_colored_images(self, img_array):
        # crop.
        if self.crop:
            img_array = self.crop_images(img_array)
        
        # resize
        if self.dimension_shift:
            img_array = self.resize_images_random(img_array)
        else:
            img_array = self.resize_images(img_array)
        height, width = img_array.shape[:2]
        
        # bright.
        if self.brightness:
            img_array = cv2.LUT(img_array, random.choice(self.brightness_tables))  #brightness adjustment
        
        img_array = img_array.astype(np.uint16)
        # gray.
        if self.gray and np.random.binomial(1, 0.05) == 1:
            img_array = self.gray_scale(img_array)   
        
        # add gaussian and speckle noise
        if self.blurring:
            img_array = self.add_colored_blur(img_array, height, width)   
                       
        if self.spots:
            img_array = self.add_colored_spots(img_array, height, width)
         
        # add padding
        img_array = self.pad_images(img_array, height, width) 
        img_array = np.clip(img_array, 0 , 255)      

        #flip or not
        if self.transformation:
            if random.choice(self.flip):
                img_array = cv2.flip(img_array, 1) 
                
            #rotate 0, 90, 180, or 270 degrees
            #img_array = img_array.astype(np.float64)
            img_array = cv2.warpAffine(img_array, 
                                       random.choice(self.rotation_matrix), 
                                       (C.work_image_size, C.work_image_size))
        return img_array
              
    def augment_gray_images(self):
        pass
          
    """below are the function that modify the images"""
    
    """resize manipulations"""
    def resize_images(self, img_array):
        height, width = img_array.shape[:2]
        if height > width:
            img_array = cv2.resize(img_array, 
                                   (C.work_image_size * width // height, 
                                    C.work_image_size))
        else:
            img_array = cv2.resize(img_array, 
                                   (C.work_image_size,
                                    C.work_image_size * height // width))
        return img_array              

    def resize_images_random(self, img_array):
        height, width = img_array.shape[:2]
        if height > width:
            random_width = np.random.randint(C.work_image_size * width // (height * self.max_shrink_ratio),
                                             high = C.work_image_size)
            img_array = cv2.resize(img_array, (random_width, C.work_image_size))
        else:
            random_height = np.random.randint(C.work_image_size * height // (width * self.max_shrink_ratio),
                                              high = C.work_image_size)
            img_array = cv2.resize(img_array, (C.work_image_size, random_height))
        return img_array         
        
    """crop manipulations"""
    def crop_images(self, img_array):
        # test ratio, find size range
        height, width = img_array.shape[:2]        
        if width / height >= self.ratio_threshold:
            '''
            If image is too wide, the random crop has the same height with
            random width that is at least the length of height
            '''
            cropped_height = height   
            cropped_width = np.random.randint(height, high = width)
            top = 0
            left = np.random.randint(0, high = width - cropped_width)
            
        elif height / width >= self.ratio_threshold:
            # reversed for image that is too tall
            cropped_height = np.random.randint(width, high = height)
            cropped_width = width
            top = np.random.randint(0, high = height - cropped_height)
            left = 0
            
        elif width > height:
            # non-extreme wide images
            cropped_height = np.random.randint(width // self.ratio_threshold, 
                                            high = height)
            cropped_width = np.random.randint(width // self.ratio_threshold, 
                                            high = width)   
            top = np.random.randint(0, high = height - cropped_height)
            left = np.random.randint(0, high = width - cropped_width)            
            
        elif height >= width:
            # non-extreme tall images
            cropped_height = np.random.randint(height // self.ratio_threshold, 
                                            high = height)
            cropped_width = np.random.randint(height // self.ratio_threshold, 
                                            high = width)    
            top = np.random.randint(0, high = height - cropped_height)
            left = np.random.randint(0, high = width - cropped_width)

        # crop image
        return img_array[top:top+cropped_height, left:left+cropped_width, :]
    
    """padding manipulations"""   
    def pad_images(self, img_array, height, width):
        difference = height - width
        if self.transformation:
            random_pad = np.random.randint(3)
        else:
            random_pad = 1
        if difference > 0:
            background = self.create_background(random_pad)
            if self.transformation:
                left = np.random.randint(0, high = difference)
            else:
                left = difference // 2
            background[:, left:left+width, :] = img_array
        elif difference < 0:
            background = self.create_background(random_pad)
            if self.transformation:
                top = np.random.randint(0, high = -difference)
            else:
                top = -difference // 2
            background[top:top+height, :, :] = img_array
        else:
            background = img_array
        return background
    
    def create_background(self, rand_value):
        if rand_value == 0:
            background = np.random.randint(256, 
                                           size = (C.work_image_size, C.work_image_size, C.channels),
                                           dtype = np.uint16)
        if rand_value == 1:
            background = np.full((C.work_image_size, C.work_image_size, C.channels), 
                                 0, dtype = np.uint16)         
        if rand_value == 2:
            background = np.full((C.work_image_size, C.work_image_size, C.channels), 
                                 255, dtype = np.uint16)  
        return background   
    
    """pixle color manipulations"""        
    def add_colored_blur(self, img_array, height, width):
        # gaussian noise
        gauss = np.random.normal(
                0, random.choice(self.gauss_variance), 
                (height, width, C.channels)
                )
        img_array = img_array + gauss
        img_array = np.clip(img_array, 0, 255)   
        return img_array
        
    def add_colored_spots(self, img_array, height, width):
        # salt and pepper noise, 6 layers because there is ceiling and floor  
        mask = np.random.binomial(  
                1, random.choice(self.spot_probability), 
                size=(height, width, C.channels * 2)
                ) 
        mask[mask == 1] = 255
        img_array += mask[:,:,0:C.channels]
        img_array = np.clip(img_array, 0, 255)
        img_array -= mask[:,:,C.channels:C.channels * 2]
        img_array = np.clip(img_array, 0, 255)        
        return img_array
    
    def LUT_table(self, brightness_value):
        # create a table for cv2.LUT
        invGamma = 1.0 / brightness_value
        table = np.zeros(256, dtype="uint8")
        for i in np.arange(0,256):
            table[i] = ((i/255.0)**invGamma)*255
           
        table = np.dstack((table, table, table))
        return table  
  
    def gray_scale(self, img_array):
        img_array = np.sum(img_array, axis = -1) // C.channels
        return np.stack((img_array, img_array, img_array), axis = 2)

