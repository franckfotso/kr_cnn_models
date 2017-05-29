# --------------------------------------------------------
# Written by Romyny
# License MIT
# --------------------------------------------------------
# Usage:
# python finetune_net --network alexnet --model cifar10_alexnet_fintuned.hdf5 \
#    --dropout 1 --batch-size 50 --epochs 10
#
# python finetune_net --network vgg16 --model cifar10_vgg16_fintuned.hdf5 \
# --dropout 1 --batch-size 5 --epochs 10
#
# python finetune_net --network vgg19 --model cifar10_vgg19_fintuned.hdf5 \
# --dropout 1 --batch-size 3 --epochs 10


from __future__ import print_function
from libs.cnn.convnetfactory import ConvNetFactory
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--network", required=True, help="name of network to build")
ap.add_argument("-m", "--model", required=True, help="path to output model file")
ap.add_argument("-d", "--dropout", type=int, default=-1,
                help="whether or not dropout should be used")
ap.add_argument("-f", "--activation", type=str, default="tanh",
                help="activation function to use (LeNet only)")
ap.add_argument("-e", "--epochs", type=int, default=20, help="# of epochs")
ap.add_argument("-b", "--batch-size", type=int, default=32,
                help="size of mini-batches passed to network")
ap.add_argument("-v", "--verbose", type=int, default=1,
                help="verbosity level")
args = vars(ap.parse_args())

print("[INFO] network: {}".format(args["network"]))
print("[INFO] batch_size: {}".format(args["batch_size"]))

learningRate = 0.001
weightDecay = 1e-6
momentum = 0.9
numClasses = 10
(imgRows, imgCols) = [0,0]
numChannels = 3

if args["network"] == "alexnet":
    (imgRows, imgCols) = [227, 227]
    weightDecay = 0.0005
elif args["network"] in ["vgg16","vgg19"]:
    (imgRows, imgCols) = [224, 224]
    weightDecay = 0.0005
else:
    raise("[ERROR] Unknown network")

# data generator
train_datagen = ImageDataGenerator(rescale=1./255)
#train_datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32) # mean imageNet

val_datagen = ImageDataGenerator(rescale=1./255)
#val_datagen.mean=np.array([103.939, 116.779, 123.68],dtype=np.float32) # mean imageNet

batch_size_gen = args["batch_size"]
    
# this generator that will read pictures found in
# sub-folders of 'data/cifar10/train', and generate batches
print("[INFO] loading train_data via train_generator...")
train_generator = train_datagen.flow_from_directory(
    'data/cifar10/train',  # target directory of images ordered by sub-folders
    target_size=(imgRows, imgCols),  # resize images to 227x227 | 224x224
    batch_size=batch_size_gen, # we use a batch size of 50
    class_mode='categorical', # we use binary labels
   )
    
# this generator that will read pictures found in
# sub-folders of 'data/cifar10/test', and generate batches
print("[INFO] loading test_data via test_generator...")
val_generator = val_datagen.flow_from_directory(
    'data/cifar10/test',  # target directory of images ordered by sub-folders
    target_size=(imgRows, imgCols),  # resize images to 227x227 | 224x224
    batch_size=batch_size_gen, # we use a batch size of 50
    class_mode='categorical')  # we use binary labels

# load pretrained model & extract the bottleneck features
print("[INFO] load pretrained model...")
model = None
if args["network"] == "vgg16":
    base_model = VGG16(weights='imagenet', include_top=False, classes=numClasses, 
                       input_tensor=Input(shape=(imgRows, imgCols, numChannels)))
    # freeze all layes involved in pre-trained model
    #for layer in base_model.layers:
        #layer.trainable = False
    # add to top layers to train
    flatten = Flatten()(base_model.output)
    fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
    fc6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
    fc7 = Dropout(0.5)(fc7)
    fc8 = Dense(numClasses,name='fc8')(fc7)      
    softmax = Activation("softmax",name="softmax")(fc8)
    
    model = Model(input=base_model.input, output=softmax)
        
elif args["network"] == "vgg19":
    base_model = VGG19(weights='imagenet', include_top=False, classes=numClasses, 
                       input_tensor=Input(shape=(imgRows, imgCols, numChannels)))
    # freeze all layes involved in pre-trained model
    #for layer in base_model.layers:
        #layer.trainable = False
    # add to top layers to train
    flatten = Flatten()(base_model.output)
    fc6 = Dense(4096, activation='relu',name='fc6')(flatten)
    fc6 = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation='relu',name='fc7')(fc6)
    fc7 = Dropout(0.5)(fc7)
    fc8 = Dense(numClasses,name='fc8')(fc7)      
    softmax = Activation("softmax",name="softmax")(fc8)
    
    model = Model(input=base_model.input, output=softmax)
elif args["network"] == "alexnet":
    kargs = {"dropout": args["dropout"] > 0, "activation": args["activation"], "include_top": False}
    model = ConvNetFactory.build(args["network"], 3, imgRows, imgCols, numClasses, **kargs)
else:
    raise("[ERROR] Unknown network")

print("[INFO] compiling model...") 
sgd = SGD(lr=learningRate, decay=weightDecay, momentum=momentum, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# start the training process
print("[INFO] starting training...")
model.fit_generator(
        train_generator,
        samples_per_epoch=41129, # samples for cifar10 train_data # 41129
        nb_epoch=args["epochs"],
        verbose=args["verbose"])

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate_generator(val_generator, 
                                            val_samples=10000 # samples for cifar10 test_data #10000
                                            )
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save_weights(args["model"]) 
