from tensorflow.image import resize
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Flatten, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Activation, AveragePooling2D
from tensorflow.keras.layers import Reshape, multiply, Lambda, Dropout, LeakyReLU
from tensorflow.keras.backend import int_shape, print_tensor

def model_head(x): # 128 x 128 x 3 
    # layer_name contruction 
    layer_name = 'head_{}_{}'
    idx = 1
    # stem reduction
    x = conv_module(x, 32, 3, 
                    layer_name.format(idx, '{}'),
                    strides = 2, 
                    padding = 'valid') # 63 x 63 x 32
    idx += 1
    x = conv_module(x, 32, 3, layer_name.format(idx, '{}'))
    idx += 1
    x = conv_module(x, 64, 3, layer_name.format(idx, '{}'))
    idx += 1
    
    # stem inception block 1  
    branch_1 = MaxPooling2D(3, strides = 2, padding = 'valid', 
                            name = layer_name.format(str(idx) + 'b1', 'maxPool'))(x) 
    branch_2 = conv_module(x, 64, 3, 
                           layer_name.format(str(idx) + 'b2', '{}'), 
                           strides = 2, padding = 'valid')
    x = Concatenate()([branch_1, branch_2])  # 31 x 31 x 128
    
    # set inception block 2
    idx += 1
    x = squeeze_excite(x, layer_name.format(idx, '{}'))
    
    idx += 1
    branch_1 = conv_module(x, 64, 1, layer_name.format(str(idx) + 'b1', '{}')) 
    
    branch_2 = conv_module(x, 64, 1, layer_name.format(str(idx) + 'b2l1', '{}'))
    branch_2 = conv_module(branch_1, 64, 5, layer_name.format(str(idx) + 'b2l2', '{}'))   
    
    branch_3 = conv_module(x, 64, 1, layer_name.format(str(idx) + 'b3l1', '{}'))
    branch_3 = conv_module(branch_3, 64, 3, layer_name.format(str(idx) + 'b3l2', '{}'))
    branch_3 = conv_module(branch_3, 64, 3, layer_name.format(str(idx) + 'b3l3', '{}'))  
    
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same',
                                   name = layer_name.format(str(idx) + 'b4', 'convPool'))(x)
    branch_pool = conv_module(branch_pool, 64, 1, layer_name.format(str(idx) + 'b4', '{}'))    
    
    x = Concatenate()([branch_1, branch_2, branch_3, branch_pool])   # 31 x 31 x 256
    idx += 1
    return squeeze_excite(x, layer_name.format(idx, '{}'))
    
def model_main(x):       
    # block A
    for i in range(1):
        if i == 0:  # split to residual          
            x_r = inception_residual_A(x, i+1) 
        else:
            x_r = inception_residual_A(x_r, i+1) 
    x = Lambda(residual_sum, arguments = {'scale' : 1.0}, name = 'blockA_res_sum')([x, x_r])    
    x = inception_residual_AR(x)
    
    # block B
    for i in range(3):
        if i == 0:   # split to residual  
            x_r = inception_residual_B(x, i+1)
        else:
            x_r = inception_residual_B(x_r, i+1)
    x = Lambda(residual_sum, arguments = {'scale' : 1.0}, name = 'blockB_res_sum')([x, x_r])
    x = inception_residual_BR(x)
    
    # block C
    for i in range(3):
        if i == 0:  # split to residual
            x_r = inception_residual_C(x, i+1)
        else:
            x_r = inception_residual_C(x_r, i+1)
            
    return Lambda(residual_sum, arguments = {'scale' : 1.0}, name = 'blockC_res_sum')([x, x_r])

def model_tail(x, class_num):
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x) 
#    x = Dropout(0.2)(x)
    return Dense(2, activation = "softmax")(x)     
       
def conv_module(x,
                filters, 
                kernel_size,
                layer_name,
                strides = 1,
                padding = 'same',
                activation = "relu",
                use_bias = False):
    x = Conv2D(filters, 
               kernel_size, 
               strides = strides, 
               padding = padding, 
               use_bias = use_bias,
               name = layer_name.format('conv'))(x)
    if use_bias == False:
        x = BatchNormalization(name = layer_name.format('convNorm'))(x)
    if activation != None:
        x = Activation(activation, name = layer_name.format('convActive'))(x)
    return x 

def inception_residual_A(x, idx):   #block A    before first round: 31 x 31 x 256
    layer_name = 'blockA' + str(idx) + '_{}_{}'
    branch_1 = conv_module(x, 96, 1, layer_name.format('1b1', '{}'))    
    
    branch_2 = conv_module(x, 64, 1, layer_name.format('1b2l1', '{}'))
    branch_2 = conv_module(branch_2, 96, 3, layer_name.format('1b2l2', '{}'))  
     
    branch_3 = conv_module(x, 64, 1, layer_name.format('1b3l1', '{}'))
    branch_3 = conv_module(branch_3, 64, 3, layer_name.format('1b3l2', '{}'))
    branch_3 = conv_module(branch_3, 96, 3, layer_name.format('1b3l3', '{}'))   
    
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same',
                                   name = layer_name.format('1b4', 'convPool'))(x)
    branch_pool = conv_module(branch_pool, 96, 1, layer_name.format('1b4', '{}'))        

    mix = Concatenate()([branch_1, branch_2, branch_3, branch_pool])   # 31 x 31 x 384
    
    mix = conv_module(mix, int_shape(x)[-1], 1, 
                      layer_name.format('1', '{}'), 
                      activation = None)    
    return squeeze_excite(mix, layer_name.format('1', '{}')) # 31 x 31 x 256
    
def inception_residual_AR(x):  #block A reduction   31 x 31 x 256
    layer_name = 'blockAR_{}_{}'
    branch_0 = conv_module(x, 256, 3, 
                           layer_name.format('1b1', '{}'),
                           strides = 2, padding = 'valid')      
    branch_pool = MaxPooling2D(3, strides = 2, padding = 'valid',
                               name = layer_name.format('1b2', 'maxPool'))(x)  
    mixed = Concatenate()([branch_0, branch_pool])    # 15 x 15 x 512    
    return squeeze_excite(mixed, layer_name.format('1', '{}'))
        
def inception_residual_B(x, idx):  #block B       15 x 15 x 512  
    layer_name = 'blockB' + str(idx) + '_{}_{}'
    branch_1 = conv_module(x, 256, 1, layer_name.format('1b1', '{}')) 
    
    branch_2 = conv_module(x, 192, 1, layer_name.format('1b2l1', '{}'))
    branch_2 = conv_module(branch_2, 192, [1, 7], layer_name.format('1b2l2', '{}'))   
    branch_2 = conv_module(branch_2, 192, [7, 1], layer_name.format('1b2l3', '{}')) 
    
    branch_3 = conv_module(x, 192, 1, layer_name.format('1b3l1', '{}'))
    branch_3 = conv_module(branch_3, 192, [1, 7], layer_name.format('1b3l2', '{}'))
    branch_3 = conv_module(branch_3, 192, [7, 1], layer_name.format('1b3l3', '{}'))
    branch_3 = conv_module(branch_3, 192, [1, 7], layer_name.format('1b3l4', '{}'))
    branch_3 = conv_module(branch_3, 192, [7, 1], layer_name.format('1b3l5', '{}')) 
        
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same',
                                   name = layer_name.format('1b4l1', 'convPool'))(x)
    branch_pool = conv_module(branch_pool, 128, 1, layer_name.format('1b4l2', '{}'))        
    
    mix = Concatenate()([branch_1, branch_2, branch_3, branch_pool])  # 15 x 15 x 768 
    mix = conv_module(mix, int_shape(x)[-1], 1,
                      layer_name.format('1', '{}'), 
                      activation = None)  # 15 x 15 x 512 
    return squeeze_excite(mix, layer_name.format('1', '{}'))
    
def inception_residual_BR(x):  #block B reduction    15 x 15 x 512 
    layer_name = 'blockBR_{}_{}'
    branch_1 = conv_module(x, 192, 1, layer_name.format('1b1l1', '{}'))
    branch_1 = conv_module(branch_1, 192, 3, 
                           layer_name.format('1b1l2', '{}'),
                           strides = 2, padding = 'valid')  
    
    branch_2 = conv_module(x, 256, 1, layer_name.format('1b2l1', '{}'))
    branch_2 = conv_module(branch_2, 256, [1, 7], layer_name.format('1b2l2', '{}'))
    branch_2 = conv_module(branch_2, 320, [7, 1], layer_name.format('1b2l3', '{}'))
    branch_2 = conv_module(branch_2, 320, 3, 
                           layer_name.format('1b2l4', '{}'),
                           strides = 2, padding = 'valid')
    
    branch_pool = MaxPooling2D(3, strides = 2, padding = 'valid',
                               name = layer_name.format('1b3', 'maxPool'))(x)  
    
    branches = [branch_1, branch_2, branch_pool]    # 7 x 7 x 1024
    mixed = Concatenate()(branches)   
    return squeeze_excite(mixed, layer_name.format('1', '{}'))    
    
        
def inception_residual_C(x, idx):  #block C    7 x 7 x 1024
    layer_name = 'blockC' + str(idx) + '_{}_{}'
    branch_1 = conv_module(x, 256, 1, layer_name.format('1b1', '{}'))
    
    branch_2 = conv_module(x, 256, 1, layer_name.format('1b2l1', '{}'))
    branch_3 = conv_module(branch_2, 256, [1, 3], layer_name.format('1b3', '{}'))
    branch_2 = conv_module(branch_2, 256, [3, 1], layer_name.format('1b2l2', '{}'))
    
    branch_4 = conv_module(x, 256, 1, layer_name.format('1b4l1', '{}'))
    branch_4 = conv_module(branch_4, 320, [1, 3], layer_name.format('1b4l2', '{}'))
    branch_4 = conv_module(branch_4, 384, [3, 1], layer_name.format('1b4l3', '{}'))   
    branch_5 = conv_module(branch_4, 256, [1, 3], layer_name.format('1b5', '{}'))
    branch_4 = conv_module(branch_4, 256, [3, 1], layer_name.format('1b4l4', '{}')) 
        
    branch_pool = AveragePooling2D(3, strides = 1, padding = 'same',
                                   name = layer_name.format('1b6l1', 'convPool'))(x)
    branch_pool = conv_module(branch_pool, 256, 1, layer_name.format('1b6l2', '{}'))
    
    branches = [branch_1, branch_2, branch_3, branch_4, branch_5, branch_pool] # 7 x 7 x 1536
    mix = Concatenate()(branches)   
    mix = conv_module(mix, int_shape(x)[-1], 1, 
                      layer_name.format('1', '{}'),
                      activation = None) # 7 x 7 x 1024
    return squeeze_excite(mix, layer_name.format('1', '{}'))
  
def squeeze_excite(x, layer_name, ratio=16):
    filters = int_shape(x)[-1]
    se = GlobalAveragePooling2D(name = layer_name.format('sePool'))(x)
    se = Flatten(name = layer_name.format('seFlat'))(se)
    se = Dense(filters // ratio, 
               activation='relu', 
               kernel_initializer = 'he_normal', 
               use_bias = False,
               name = layer_name.format('dense1'))(se)
    se = Dense(filters, 
               activation='sigmoid', 
               kernel_initializer = 'he_normal', 
               use_bias = False,
               name = layer_name.format('dense2'))(se)
    return multiply([x, se])        
    
def residual_sum(inputs, scale = 0.3):
    return inputs[0] + inputs[1] * scale

def resize_and_normalize(x, size = (256, 256)):
    x = resize(x, size)
    return x / 255 - 0.5

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
