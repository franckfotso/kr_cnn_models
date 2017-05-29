# --------------------------------------------------------
# Written by Romyny
# License MIT
# --------------------------------------------------------
# usage:
# python train_basicnet.py --network shallownet --model output/cifar10_shallownet.hdf5 --epochs 20
# python train_basicnet.py --network lenet --model output/cifar10_lenet.hdf5 --epochs 20
#
# python train_basicnet.py --network karpathynet --model output/cifar10_karparthynet_without_dropout.hdf5 --epochs 100
# python train_basicnet.py --network karpathynet --model output/cifar10_karparthynet_with_dropout.hdf5 --dropout 1 --epochs 100
#
# python train_basicnet.py --network minivggnet --model output/cifar10_minivggnet_without_dropout.hdf5 --epochs 200
# python train_basicnet.py --network minivggnet --model output/cifar10_minivggnet_with_dropout.hdf5 --dropout 1 --epochs 200


from __future__ import print_function
from libs.cnn.convnetfactory import ConvNetFactory
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
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

# load the training and testing data
print("[INFO] loading training data...")
((trainData, trainLabels), (testData, testLabels)) = cifar10.load_data()
# scale data into the range [0, 1], other option: mean-center (see esri)
trainData = trainData.astype("float") / 255.0
testData = testData.astype("float") / 255.0

# transform the training and testing labels into vectors [0, numClasses]
trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# collect the keyword arguments to the network
kargs = {"dropout": args["dropout"] > 0, "activation": args["activation"]}

# train the model using SGD
print("[INFO] compiling model...")
model = ConvNetFactory.build(args["network"], 3, 32, 32, 10, **kargs)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# start the training process
print("[INFO] starting training...")
model.fit(trainData, trainLabels, batch_size=args["batch_size"],
          nb_epoch=args["epochs"], verbose=args["verbose"])

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(testData, testLabels,
                                  batch_size=args["batch_size"], verbose=args["verbose"])
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])
