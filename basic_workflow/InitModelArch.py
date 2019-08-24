def ModelTrainer(save_model = False):
    import os, time
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
    
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Input

    #from tensorflow.keras.backend import cast, set_floatx, set_epsilon, cast_to_floatx
    from tensorflow.keras.callbacks import TensorBoard
    from Validation import Validation
    from ImageAugmentation import ImageAugmentation
    from InputProcess import InputProcess
    from DataFeeder import DataGenerator
    from ModelArch import ModelHead, ModelMain, ModelTail

    #NAME = "CatDog-ColorGenData-SeInRes11-HighScale{}".format(int(time.time()))
    NAME = "TrialRun1"
    
    tensorboard = TensorBoard(log_dir = 'log/{}'.format(NAME))
    #tf.config.gpu.set_per_process_memory_fraction(1)
    #tf.config.gpu.set_per_process_memory_growth(True)
    
    GPUsetting = tf.GPUOptions(per_process_gpu_memory_fraction = 1, allow_growth = True)
    sess = tf.Session(config = tf.ConfigProto(gpu_options = GPUsetting))
    
    main_storage = "data_folder"
    image_size = 128
    Categories = ["Dog", "Cat"]
    
    InputProcess = InputProcess(main_storage, image_size, Categories)
    Augmentation = ImageAugmentation(InputProcess)
    
    #Augmentation  = ImageAugmentation(
    #        InputProcess, Brightness = False, Gray = False, Transformations = False, 
    #        Blurring = False, Spots = False
    #        )
    
    Validation = Validation(InputProcess)
    X_test, y_test = Validation.DeployImages()
    X_test = X_test/255 - 0.5
    
    batch_size = 64
    epoch_step = 20000
    tensor_shape = (batch_size, image_size, image_size, InputProcess.channels)
    #set_floatx("float32")
    #set_epsilon(1e-4)   
    
    Opt = Adam(lr = 0.0007)
    
    Generator = DataGenerator(InputProcess, Augmentation, batch_size, epoch_step)
    
    #model structure goes here
    input_matrix = Input(tensor_shape[1:])
    x = ModelHead(input_matrix)
    x = ModelMain(x)
    output_matrix = ModelTail(x)
    
    model = Model(inputs = input_matrix, outputs = output_matrix)
    model.compile(
            loss = "binary_crossentropy", optimizer = Opt, metrics = ["accuracy"]
            )
    
    model.fit_generator(
            Generator, validation_data = (X_test, y_test),
            use_multiprocessing=True, callbacks = [tensorboard],
            epochs = 250, workers = 7 
            )
    
    if save_model:
        model.save(NAME+".h5")










