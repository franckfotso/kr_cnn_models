# --------------------------------------------------------
# Written by: Romuald FOTSO
# Licensed: MIT License
# Copyright (c) 2017
# --------------------------------------------------------
# usage:
# python train_deepnet.py --network alexnet --model output/cifar10_alexnet.hdf5 --dropout 1 --batch-size 50 --epochs 35
# python train_deepnet.py --network vgg16 --model output/cifar10_vgg16.hdf5 --dropout 1 --batch-size 50 --epochs 35


from __future__ import print_function
from libs.cnn.convnetfactory import ConvNetFactory
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
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

weightDecay = 1e-6
numClasses = 10
(imgRows, imgCols) = [0,0]

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
val_datagen = ImageDataGenerator(rescale=1./255)
batch_size_gen = args["batch_size"]
    
# this generator that will read pictures found in
# sub-folders of 'data/cifar10/train', and generate batches
print("[INFO] loading train_data via train_generator...")
train_generator = train_datagen.flow_from_directory(
    'data/cifar10/train',  # target directory of images ordered by sub-folders
    target_size=(imgRows, imgCols),  # resize images to 227x227
    batch_size=batch_size_gen, # we use a batch size of 50
    class_mode='categorical', # we use binary labels
   )  
    
# this generator that will read pictures found in
# sub-folders of 'data/cifar10/test', and generate batches
print("[INFO] loading test_data via test_generator...")
val_generator = val_datagen.flow_from_directory(
    'data/cifar10/test',  # target directory of images ordered by sub-folders
    target_size=(imgRows, imgCols),  # resize images to 227x227
    batch_size=batch_size_gen, # we use a batch size of 50
    class_mode='categorical')  # we use binary labels

# collect the keyword arguments to the network
kargs = {"dropout": args["dropout"] > 0, "activation": args["activation"], "include_top": True}

# train the model using SGD
print("[INFO] compiling model...")    
model = ConvNetFactory.build(args["network"], 3, imgRows, imgCols, numClasses, **kargs)
sgd = SGD(lr=0.01, decay=weightDecay, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# start the training process
print("[INFO] starting training...")
model.fit_generator(
        train_generator,
        samples_per_epoch=41129, # samples for cifar10 train_data
        nb_epoch=args["epochs"],
        verbose=args["verbose"])

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate_generator(val_generator, 
                                            val_samples=10000 # samples for cifar10 test_data
                                            )
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save_weights(args["model"]) 
