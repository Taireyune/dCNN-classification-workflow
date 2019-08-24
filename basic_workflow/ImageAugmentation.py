import numpy as np
import cv2, random
import matplotlib.pyplot as plt
class ImageAugmentation:
    def __init__(
            self, InputProcess, Brightness = True, Gray = True, 
            Transformations = True, Blurring = True, Spots = True, 
            ):
        self.image_size = InputProcess.image_size
        self.channels = InputProcess.channels
        is_colored = InputProcess.is_colored
        # initialize gray.
        self.Gray = Gray
        
        # initialize padding, rotating and flipping
        self.Transformations = Transformations
        if Transformations:
            self.pad = [-1, 0, 1]
        else:
            self.pad = [0]
        self.flip = [0,1]
        self.rotationMatrix = []        
        for i in range(4):
            matrix = cv2.getRotationMatrix2D(
                    (self.image_size/2, self.image_size/2), 90 * i, 1
                    )        
            self.rotationMatrix.append(matrix)

        #create a list of brightness tables
        self.Brightness = Brightness    
        self.brightness_tables = []
        for brightness_value in [0.5, 1, 1.5]:
            table  = self.LUTtable(brightness_value, is_colored)
            self.brightness_tables.append(table)
      
        # setup channel for gaussian noise
        self.Blurring = Blurring
        self.gauss_variance = [0, 0, 0, 0, 0, 0, 5, 10]
        
        # salt and peppering
        self.Spots = Spots
        if is_colored:
            self.spot_probability = [0, 0, 0, 0, 0.015, 0.03]          
        else:              
            self.spot_probability = [0, 0, 0, 0.1, 0.2]

        
    "Execute this to start the run"
    def AugmentColoredImages(self, img_array):   
        height, width = img_array.shape[:2]
        # bright.
        img_array = cv2.LUT(img_array, random.choice(self.brightness_tables))  #brightness adjustment
        
        # gray.
        if self.Gray and np.random.binomial(1, 0.05) == 1:
            img_array = self.GrayScale(img_array)   

        # add gaussian and speckle noise
        if self.Blurring:
            img_array = self.AddColoredBlur(img_array, height, width)                    

        if self.Spots:
            img_array = self.AddColoredSpots(img_array, height, width)
        
        # add padding
        img_array = self.PadImages(img_array, height, width) 
        img_array = np.clip(img_array, 0 , 255)      
        
        #flip or not
        if self.Transformations:
            if random.choice(self.flip):
                img_array = cv2.flip(img_array, 1) 
                
            #rotate 0, 90, 180, or 270 degrees
            img_array = img_array.astype(float)
            img_array = cv2.warpAffine(
                    img_array, random.choice(self.rotationMatrix), 
                    (self.image_size, self.image_size)
                    )

        return img_array
              
    def AugmentGrayImages(self):
        pass
          
    """below are the function that modify the images"""
           
    def PadImages(self, img_array, height, width):     
        difference = height - width
        position = random.choice(self.pad)
        
        if difference > 0:
            pad1 = int(difference/2 + 0.5)
            pad2 = difference - pad1
            if position == 0:
                img_array = self.AddPadding(img_array, 0, 0, pad1, pad2)
            elif position == -1:
                img_array = self.AddPadding(img_array, 0, 0, 0, difference)
            elif position == 1:
                img_array = self.AddPadding(img_array, 0, 0, difference, 0)         
            
        elif difference < 0:    
            pad1 = int(-difference/2 + 0.5)
            pad2 = -difference - pad1
            if position == 0:   
                img_array = self.AddPadding(img_array, pad1, pad2, 0, 0)
            elif position == -1:                
                img_array = self.AddPadding(img_array, -difference, 0, 0, 0)
            elif position == 1:
                img_array = self.AddPadding(img_array, 0, -difference, 0, 0) 
                
        return img_array
        
    def AddPadding(self, img_array, top, bottom, left, right):
        chooser = np.random.randint(3, size = 4)
        height, width = img_array.shape[:2]
        
        top_pad = self.CreatePadding(chooser[0], top, width)            
        bottom_pad = self.CreatePadding(chooser[1], bottom, width)
        img_array = np.concatenate((top_pad, img_array, bottom_pad), axis = 0)
        
        height, width = img_array.shape[:2]
        
        left_pad = self.CreatePadding(chooser[2], height, left)
        right_pad = self.CreatePadding(chooser[3], height, right)
        img_array = np.concatenate((left_pad, img_array, right_pad), axis = 1)
        
        return img_array

    def CreatePadding(self, rand_value, height, width):
        if rand_value == 0:
            pad = np.random.randint(256, size = (height, width, self.channels))
        if rand_value == 1:
            pad = np.full((height, width, self.channels), 0, dtype = int)         
        if rand_value == 2:
            pad = np.full((height, width, self.channels), 255, dtype = int)  
        return pad            
            
    def AddColoredBlur(self, img_array, height, width):
        # gaussian noise
        gauss = np.random.normal(
                0, random.choice(self.gauss_variance), 
                (height, width, self.channels)
                )
        img_array = img_array + gauss
        img_array = np.clip(img_array, 0, 255)   
        return img_array
        
    def AddColoredSpots(self, img_array, height, width):
        # salt and pepper noise, 6 layers because there is ceiling and floor  
        mask = np.random.binomial(  
                1, random.choice(self.spot_probability), 
                size=(height, width, self.channels * 2)
                ) 
        mask[mask == 1] = 255
        img_array += mask[:,:,0:self.channels]
        img_array = np.clip(img_array, 0, 255)
        img_array -= mask[:,:,self.channels:self.channels * 2]
        img_array = np.clip(img_array, 0, 255)        
        return img_array
    
    def LUTtable(self, brightness_value, is_colored):
        # create a table for cv2.LUT
        invGamma = 1.0 / brightness_value
        table = np.zeros(256, dtype="uint8")
        for i in np.arange(0,256):
            table[i] = ((i/255.0)**invGamma)*255
        
        if is_colored:     
            table = np.dstack((table, table, table))
        return table  
  
    def GrayScale(self, img_array):
        img_array = np.sum(img_array, axis = -1)/self.channels
        img_array = np.stack((img_array, img_array, img_array), axis = 2)
        return img_array









