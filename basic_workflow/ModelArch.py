import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D
from tensorflow.keras.layers import Reshape, multiply, Lambda, Dropout, LeakyReLU
from tensorflow.keras.backend import int_shape, print_tensor

def ModelHead(x): # 128 x 128 x 3 
    #stem reduction
    x = ConvModule(x, 32, 3, strides = 2, padding = 'valid') # 63 x 63 x 32
    x = ConvModule(x, 32, 3)
    x = ConvModule(x, 64, 3)
    
    branch_0 = MaxPooling2D(3, strides = 2, padding = 'valid')(x) 
    branch_1 = ConvModule(x, 64, 3, strides = 2, padding = 'valid')
    x = Concatenate()([branch_0, branch_1])  # 31 x 31 x 128
    x = SqueezeExcite(x)
    
    #stem inception block
    branch_0 = ConvModule(x, 64, 1) 
    
    branch_1 = ConvModule(x, 64, 1)
    branch_1 = ConvModule(branch_1, 64, 5)   
    
    branch_2 = ConvModule(x, 64, 1)
    branch_2 = ConvModule(branch_2, 64, 3)
    branch_2 = ConvModule(branch_2, 64, 3)  
    
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same')(x)
    branch_pool = ConvModule(branch_pool, 64, 1)    
    
    x = Concatenate()([branch_0, branch_1, branch_2, branch_pool])   # 31 x 31 x 256
    return SqueezeExcite(x)
    

def ModelMain(x):       #2, 4, 2  
    for i in range(1):
        x = InceptionResidualA(x)   
    x = InceptionResidualAR(x)
    for i in range(3):
        x = InceptionResidualB(x)
    x = InceptionResidualBR(x)
    for i in range(3):
        x = InceptionResidualC(x)
    return x

def ModelTail(x):
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x) 
    x = Dropout(0.2)(x)
    return Dense(1, activation = "sigmoid")(x)     
       
def ConvModule(x,
               filters, 
               kernel_size,
               strides = 1,
               padding = 'same',
               activation = "relu",
               use_bias = False):
    x = Conv2D(filters, 
               kernel_size, 
               strides = strides, 
               padding = padding, 
               use_bias = use_bias)(x)
    if use_bias == False:
        x = BatchNormalization()(x)
    if activation != None:
        x = Activation(activation)(x)
    return x 

def InceptionResidualA(x, scale = 0.8):   #block A    before first round: 31 x 31 x 256
    branch_0 = ConvModule(x, 96, 1)    
    
    branch_1 = ConvModule(x, 64, 1)
    branch_1 = ConvModule(branch_1, 96, 3)  
     
    branch_2 = ConvModule(x, 64, 1)
    branch_2 = ConvModule(branch_2, 64, 3)
    branch_2 = ConvModule(branch_2, 96, 3)   
    
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same')(x)
    branch_pool = ConvModule(branch_pool, 96, 1)        

    mixed = Concatenate()([branch_0, branch_1, branch_2, branch_pool])   # 31 x 31 x 384
    mixed = ConvModule(mixed, int_shape(x)[-1], 1, activation = None)
    
    mixed = SqueezeExcite(mixed) # 31 x 31 x 256
    x = Lambda(ResidualSum, arguments = {'scale' : scale})([x, mixed])
    return LeakyReLU()(x)
    
def InceptionResidualAR(x):  #block A reduction   31 x 31 x 256
    branch_0 = ConvModule(x, 256, 3, strides = 2, padding = 'valid')      
    branch_pool = MaxPooling2D(3, strides = 2, padding = 'valid')(x)  
    mixed = Concatenate()([branch_0, branch_pool])    # 15 x 15 x 512    
    return SqueezeExcite(mixed)
        
def InceptionResidualB(x, scale = 0.6):  #block B       15 x 15 x 512  
    branch_0 = ConvModule(x, 256, 1) 
    
    branch_1 = ConvModule(x, 192, 1)
    branch_1 = ConvModule(branch_1, 192, [1, 7])   
    branch_1 = ConvModule(branch_1, 192, [7, 1]) 
    
    branch_2 = ConvModule(x, 192, 1)
    branch_2 = ConvModule(branch_2, 192, [1, 7])
    branch_2 = ConvModule(branch_2, 192, [7, 1])
    branch_2 = ConvModule(branch_2, 192, [1, 7])
    branch_2 = ConvModule(branch_2, 192, [7, 1]) 
        
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same')(x)
    branch_pool = ConvModule(branch_pool, 128, 1)        
    
    mixed = Concatenate()([branch_0, branch_1, branch_2, branch_pool])  # 15 x 15 x 768 
    mixed = ConvModule(mixed, int_shape(x)[-1], 1, activation = None)  # 15 x 15 x 512 
    mixed = SqueezeExcite(mixed)
    x = Lambda(ResidualSum, arguments = {'scale' : scale})([x, mixed])
    return LeakyReLU()(x)
    
def InceptionResidualBR(x):  #block B reduction    15 x 15 x 512 
    branch_0 = ConvModule(x, 192, 1)
    branch_0 = ConvModule(x, 192, 3, strides = 2, padding = 'valid')  
    
    branch_1 = ConvModule(x, 256, 1)
    branch_1 = ConvModule(branch_1, 256, [1, 7])
    branch_1 = ConvModule(branch_1, 320, [7, 1])
    branch_1 = ConvModule(branch_1, 320, 3, strides = 2, padding = 'valid')
    
    branch_pool = MaxPooling2D(3, strides = 2, padding = 'valid')(x)  
    
    branches = [branch_0, branch_1, branch_pool]    # 7 x 7 x 1024
    mixed = Concatenate()(branches)   
    return SqueezeExcite(mixed)    
    
        
def InceptionResidualC(x, scale = 0.8):  #block C    7 x 7 x 1024
    branch_0 = ConvModule(x, 256, 1)
    
    branch_1b = ConvModule(x, 256, 1)
    branch_1a = ConvModule(branch_1b, 256, [1, 3])
    branch_1b = ConvModule(branch_1b, 256, [3, 1])
    
    branch_2b = ConvModule(x, 256, 1)
    branch_2b = ConvModule(branch_2b, 320, [1, 3])
    branch_2b = ConvModule(branch_2b, 384, [3, 1])   
    branch_2a = ConvModule(branch_2b, 256, [1, 3])
    branch_2b = ConvModule(branch_2b, 256, [3, 1]) 
        
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same')(x)
    branch_pool = ConvModule(branch_pool, 256, 1)
    
    branches = [branch_0, branch_1a, branch_1b, branch_2a, branch_2b, branch_pool] # 7 x 7 x 1536
    mixed = Concatenate()(branches)   
    mixed = ConvModule(mixed, int_shape(x)[-1], 1, activation = None) # 7 x 7 x 1024
    mixed = SqueezeExcite(mixed)
    x = Lambda(ResidualSum, arguments = {'scale' : scale})([x, mixed])
    return LeakyReLU()(x)
  
def SqueezeExcite(x, ratio=16):
    filters = int_shape(x)[-1]
    se = GlobalAveragePooling2D()(x)
    se = Flatten()(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer = 'he_normal', use_bias = False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer = 'he_normal', use_bias = False)(se)
    return multiply([x, se])        
    
def ResidualSum(inputs, scale = 0.3):
    return inputs[0] + inputs[1] * scale
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
