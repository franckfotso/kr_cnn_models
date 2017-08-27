# --------------------------------------------------------
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017
# --------------------------------------------------------

from keras.layers import merge, Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.models import Sequential, Model

from .customlayers import LRN2D
from .utils import splittensor


class ConvNetFactory:
    def __init__(self):
        pass        
    
    @staticmethod
    def build(name, *args, **kargs):
        # define the network (i.e., string => function) mappings
        mappings = {
            "shallownet": ConvNetFactory.ShallowNet,
            "lenet": ConvNetFactory.LeNet,
            "karpathynet": ConvNetFactory.KarpathyNet,
            "minivggnet": ConvNetFactory.MiniVGGNet,
            "alexnet": ConvNetFactory.AlexNet,
            "vgg16": ConvNetFactory.VGG16,
            "vgg19": ConvNetFactory.VGG19,
            }
        
        # grab the builder function from the mappings dictionary
        builder = mappings.get(name, None)
        
        # if no builder return None
        if builder is None:
            return None
        
        # build the network architecture
        return builder(*args, **kargs)
    
    @staticmethod
    def ShallowNet(numChannels, imgRows, imgCols, numClasses, **kwargs):
        # Arch: INPUT => CONV => RELU => FC
        
        # initialize the model
        model = Sequential()
        
        # define INPUT => CONV
        model.add(Convolution2D(32, 3, 3, border_mode="same",
            #input_shape=(numChannels, imgRows, imgCols)))
            input_shape=(imgRows, imgCols, numChannels)))
        # add RELU layer
        model.add(Activation("relu"))
        # add FC layer
        model.add(Flatten()) # flatten n-dim to 1-dim
        model.add(Dense(numClasses)) # fc layer
        # add softmax activation
        model.add(Activation("softmax")) # ie: logistic regression
        
        return model
    
    @staticmethod
    def LeNet(numChannels, imgRows, imgCols, numClasses, activation="tanh", **kwargs):
        # LeNet: INPUT => CONV => TANH => POOL => CONV => TANH => POOL => FC => TANH => FC
        
        # initialize the model
        model = Sequential()
        
        # INPUT => CONV
        model.add(Convolution2D(20, 5, 5, border_mode="same",
        input_shape=(imgRows, imgCols, numChannels)))
        # add TANH or RELU
        model.add(Activation(activation))
        # add POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # add CONV
        model.add(Convolution2D(50, 5, 5, border_mode="same"))
        # add TANH or RELU
        model.add(Activation(activation))
        # add POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # add FC
        model.add(Flatten())
        model.add(Dense(500))
        # add TANH or RELU
        model.add(Activation(activation))
        # add FC
        model.add(Dense(numClasses))
        # add softmax activation
        model.add(Activation("softmax")) # ie: logistic regression        
        
        return model

    @staticmethod
    def KarpathyNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # KarpathyNet: INPUT => (CONV => RELU => POOL => (DROPOUT?)) * 3 => SOFTMAX
        
        # initialize the model
        model = Sequential()
        
        # INPUT => CONV
        model.add(Convolution2D(16, 5, 5, border_mode="same",
                                input_shape=(imgRows, imgCols, numChannels)))
        # add RELU
        model.add(Activation("relu"))
        # add POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))        
        # add DROPOUT
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))
        
        # add CONV
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        # add RELU
        model.add(Activation("relu"))
        # add POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # add DROPOUT
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))
            
        # add CONV
        model.add(Convolution2D(20, 5, 5, border_mode="same"))
        # add RELU
        model.add(Activation("relu"))
        # add POOL
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # add DROPOUT
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))
            
        # add FC
        model.add(Flatten())
        model.add(Dense(numClasses))
        # add softmax activation
        model.add(Activation("softmax")) # ie: logistic regression 
        
        return model

    @staticmethod
    def MiniVGGNet(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # MiniVGGNet: CONV => RELU => CONV => RELU => POOL => FC => RELU => FC => SOFTMAX
        
        # initialize the model
        model = Sequential()
        
        # 1st sets
        model.add(Convolution2D(32, 3, 3, border_mode="same",
                                input_shape=(imgRows, imgCols, numChannels)))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))        
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.25))
            
        # 2nd sets
        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(64, 3, 3))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # define FC => RELU
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        # check to see if dropout should be applied to reduce overfitting
        if dropout:
            model.add(Dropout(0.5))
            
        # define soft-max classifier
        model.add(Dense(numClasses))
        model.add(Activation("softmax"))
        
        return model
    
    @staticmethod
    def AlexNet(numChannels, imgRows, imgCols, numClasses, include_top=True, **kwargs):        
        lrn = {"ALPHA": 0.0001, "BETA": 0.75, "GAMMA": 0.1}
        weights_path = "models/imagenet_weights/alexnet_weights.h5"
        
        # ConvNet: input
        inputs = Input(shape=(imgRows, imgCols, numChannels))
        #inputs = Input(shape=(numChannels, imgRows, imgCols))
        
        # ConvNet: layer 1
        conv1 = Convolution2D(96, 11, 11, subsample=(4,4), name="conv1", border_mode="same")(inputs)
        relu1 = Activation("relu",name="relu1")(conv1)
        lrn1 = LRN2D(alpha=lrn["ALPHA"], beta=lrn["BETA"])(relu1)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="pool1")(lrn1)
        
        # ConvNet: layer 2
        conv2 = ZeroPadding2D((2,2))(pool1)
        conv2_1, conv2_2 = [splittensor(ratio_split=2,id_split=i)(conv2) for i in range(2)]
        conv2_1 = Convolution2D(128, 5, 5, subsample=(1,1), name="conv2_1")(conv2_1)
        relu2_1 = Activation("relu",name="relu2_1")(conv2_1)
        conv2_2 = Convolution2D(128, 5, 5, subsample=(1,1), name="conv2_2")(conv2_2)
        relu2_2 = Activation("relu",name="relu2_2")(conv2_2)
        # merge the 2 sub-layers/branch
        relu2 = merge([relu2_1, relu2_2], mode='concat',concat_axis=1,name="relu_2")
        lrn2 = LRN2D(alpha=lrn["ALPHA"], beta=lrn["BETA"])(relu2)
        pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="pool2")(lrn2)
        
        # ConvNet: layer 3
        conv3 = ZeroPadding2D((1,1))(pool2)
        conv3 = Convolution2D(384, 3, 3, subsample=(1,1), name="conv3")(conv3)
        relu3 = Activation("relu",name="relu3")(conv3)
        lrn3 = LRN2D(alpha=lrn["ALPHA"], beta=lrn["BETA"])(relu3)
        
        # ConvNet: layer 4
        conv4 = ZeroPadding2D((1,1))(lrn3)
        conv4_1, conv4_2 = [splittensor(ratio_split=2,id_split=i)(conv4) for i in range(2)]
        conv4_1 = Convolution2D(192, 3, 3, subsample=(1,1), name="conv4_1")(conv4_1)
        relu4_1 = Activation("relu",name="relu4_1")(conv4_1)
        conv4_2 = Convolution2D(192, 3, 3, subsample=(1,1), name="conv4_2")(conv4_2)
        relu4_2 = Activation("relu",name="relu4_2")(conv4_2)
        # merge the 2 sub-layers/branch
        relu4 = merge([relu4_1, relu4_2], mode='concat',concat_axis=1,name="relu_4")
        lrn4 = LRN2D(alpha=lrn["ALPHA"], beta=lrn["BETA"])(relu4)
        
        # ConvNet: layer 5
        conv5 = ZeroPadding2D((1,1))(lrn4)
        conv5_1, conv5_2 = [splittensor(ratio_split=2,id_split=i)(conv5) for i in range(2)]
        conv5_1 = Convolution2D(128, 3, 3, subsample=(1,1), name="conv5_1")(conv5_1)
        relu5_1 = Activation("relu",name="relu5_1")(conv5_1)
        conv5_2 = Convolution2D(128, 3, 3, subsample=(1,1), name="conv5_2")(conv5_2)
        relu5_2 = Activation("relu",name="relu5_2")(conv5_2)
        # merge the 2 sub-layers/branch
        relu5 = merge([relu5_1, relu5_2], mode='concat',concat_axis=1,name="relu_5")
        lrn5 = LRN2D(alpha=lrn["ALPHA"], beta=lrn["BETA"])(relu5)
        pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name="pool5")(lrn5)
        
        # ConvNet: layer 6
        flatten = Flatten(name="flatten")(pool5)
        fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
        fc6 = Dropout(0.5)(fc6)
        
        # ConvNet: layer 7
        fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
        fc7 = Dropout(0.5)(fc7)
    
        # ConvNet: output
        output = Dense(numClasses,name='output')(fc7)
        softmax = Activation("softmax",name="softmax")(output)
        
        model = None
        if include_top:
            model = Model(input=inputs, output=softmax)
        else:
            model = Model(input=inputs, output=softmax)
            model.load_weights(weights_path)
            # freeze the 8 first layers involved in pre-trained model
            for layer in model.layers[:8]:
                layer.trainable = False
            
        if model == None:
            raise("[ERROR] no model built")        
           
        return model
    
    @staticmethod
    def VGG16(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # ConvNet: input
        inputs = Input(shape=(imgRows, imgCols, numChannels))
        
        # ConvNet: bloc 1
        #conv1_1 = ZeroPadding2D((1,1))(inputs)
        conv1_1 = Convolution2D(64, 3, 3, subsample=(1,1), activation="relu", name="conv1_1")(inputs)
        conv1_2 = ZeroPadding2D((1,1))(conv1_1)
        conv1_2 = Convolution2D(64, 3, 3, subsample=(1,1), activation="relu", name="conv1_2")(conv1_2)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv1_2)
        
        # ConvNet: bloc 2
        conv2_1 = ZeroPadding2D((1,1))(pool1)
        conv2_1 = Convolution2D(128, 3, 3, subsample=(1,1), activation="relu", name="conv2_1")(conv2_1)
        conv2_2 = ZeroPadding2D((1,1))(conv2_1)
        conv2_2 = Convolution2D(128, 3, 3, subsample=(1,1), activation="relu", name="conv2_2")(conv2_2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv2_2)
        
        # ConvNet: bloc 3
        conv3_1 = ZeroPadding2D((1,1))(pool2)
        conv3_1 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_1")(conv3_1)
        conv3_2 = ZeroPadding2D((1,1))(conv3_1)
        conv3_2 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_2")(conv3_2)
        conv3_3 = ZeroPadding2D((1,1))(conv3_2)
        conv3_3 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_3")(conv3_3)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv3_3)
        
        # ConvNet: bloc 4
        conv4_1 = ZeroPadding2D((1,1))(pool3)
        conv4_1 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_1")(conv4_1)
        conv4_2 = ZeroPadding2D((1,1))(conv4_1)
        conv4_2 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_2")(conv4_2)
        conv4_3 = ZeroPadding2D((1,1))(conv4_2)
        conv4_3 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_3")(conv4_3)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv4_3)
        
        # ConvNet: bloc 5
        conv5_1 = ZeroPadding2D((1,1))(pool4)
        conv5_1 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_1")(conv5_1)
        conv5_2 = ZeroPadding2D((1,1))(conv5_1)
        conv5_2 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_2")(conv5_2)
        conv5_3 = ZeroPadding2D((1,1))(conv5_2)
        conv5_3 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_3")(conv5_3)
        pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv5_3)
        
        # ConvNet: bloc 6
        flatten = Flatten()(pool5)
        fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
        fc6 = Dropout(0.5)(fc6)
        fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
        fc7 = Dropout(0.5)(fc7)
        fc8 = Dense(numClasses,name='fc8')(fc7)
        
        # ConvNet: output        
        softmax = Activation("softmax",name="softmax")(fc8)
        
        model = Model(input=inputs, output=softmax)     
        return model
    
    @staticmethod
    def VGG19(numChannels, imgRows, imgCols, numClasses, dropout=False, **kwargs):
        # ConvNet: input
        inputs = Input(shape=(imgRows, imgCols, numChannels))
        
        # ConvNet: bloc 1
        conv1_1 = ZeroPadding2D((1,1))(inputs)
        conv1_1 = Convolution2D(64, 3, 3, subsample=(1,1), activation="relu", name="conv1_1")(conv1_1)
        conv1_2 = ZeroPadding2D((1,1))(conv1_1)
        conv1_2 = Convolution2D(64, 3, 3, subsample=(1,1), activation="relu", name="conv1_2")(conv1_2)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv1_2)
        
        # ConvNet: bloc 2
        conv2_1 = ZeroPadding2D((1,1))(pool1)
        conv2_1 = Convolution2D(128, 3, 3, subsample=(1,1), activation="relu", name="conv2_1")(conv2_1)
        conv2_2 = ZeroPadding2D((1,1))(conv2_1)
        conv2_2 = Convolution2D(128, 3, 3, subsample=(1,1), activation="relu", name="conv2_2")(conv2_2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv2_2)
        
        # ConvNet: bloc 3
        conv3_1 = ZeroPadding2D((1,1))(pool2)
        conv3_1 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_1")(conv3_1)
        conv3_2 = ZeroPadding2D((1,1))(conv3_1)
        conv3_2 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_2")(conv3_2)
        conv3_3 = ZeroPadding2D((1,1))(conv3_2)
        conv3_3 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_3")(conv3_3)
        conv3_4 = ZeroPadding2D((1,1))(conv3_3)
        conv3_4 = Convolution2D(256, 3, 3, subsample=(1,1), activation="relu", name="conv3_4")(conv3_4)
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv3_4)
        
        # ConvNet: bloc 4
        conv4_1 = ZeroPadding2D((1,1))(pool3)
        conv4_1 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_1")(conv4_1)
        conv4_2 = ZeroPadding2D((1,1))(conv4_1)
        conv4_2 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_2")(conv4_2)
        conv4_3 = ZeroPadding2D((1,1))(conv4_2)
        conv4_3 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_3")(conv4_3)
        conv4_4 = ZeroPadding2D((1,1))(conv4_3)
        conv4_4 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv4_4")(conv4_4)
        pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv4_4)
        
        # ConvNet: bloc 5
        conv5_1 = ZeroPadding2D((1,1))(pool4)
        conv5_1 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_1")(conv5_1)
        conv5_2 = ZeroPadding2D((1,1))(conv5_1)
        conv5_2 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_2")(conv5_2)
        conv5_3 = ZeroPadding2D((1,1))(conv5_2)
        conv5_3 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_3")(conv5_3)
        conv5_4 = ZeroPadding2D((1,1))(conv5_3)
        conv5_4 = Convolution2D(512, 3, 3, subsample=(1,1), activation="relu", name="conv5_4")(conv5_4)
        pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv5_4)
        
        # ConvNet: bloc 6
        flatten = Flatten()(pool5)
        fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
        fc6 = Dropout(0.5)(fc6)
        fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
        fc7 = Dropout(0.5)(fc7)
        fc8 = Dense(numClasses,name='fc8')(fc7)
        
        # ConvNet: output        
        softmax = Activation("softmax",name="softmax")(fc8)
        
        model = Model(input=inputs, output=softmax)     
        return model
