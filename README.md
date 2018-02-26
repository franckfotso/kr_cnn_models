# kr_cnn_models
Implementation of CNN architectures with keras

This project involves the following architectures:
ShallowNet, LeNet, KarpathyNet, miniVggNet, AlexNet, vgg16 and vgg19

Created by Franck FOTSO

## Purposes:

Our main goal was to implementation some well-known CNN models with keras interface.
In addition, we aim to show the sharp difference between the shallow and deep Neural Networks.

## Datasets:

This project focus on the dataset: **CIFAR-10**
It is available (on download) with keras packages or through this link:

    https://www.cs.toronto.edu/~kriz/cifar.html
    http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

## Hardwares/Softwares:
    OS: Ubuntu 16.04 64 bit
    GPU: Nvidia GTX 950M
    Cuda 8.0

## Prerequisites:

1) Keras (for python 2.7): https://keras.io/

2) Pre-trained models (**AlexNet** & **Vgg16/19**) for finetune tasks: available (on download) with keras packages

## Installation:

1) Clone the project repository:

    $ git clone https://github.com/romyny/kr_cnn_models.git
    
2) Install Keras (according theano or tensorflow backend):

    https://keras.io/#installation or
    http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/

## Experiments:

1) Train shallow neural networks:

    * $ python tools/train_basicnet.py --network shallownet --model output/cifar10_shallownet.hdf5 --epochs 20
    * $ python tools/train_basicnet.py --network lenet --model output/cifar10_lenet.hdf5 --epochs 20
    * $ python tools/train_basicnet.py --network karpathynet --model output/cifar10_karparthynet_without_dropout.hdf5 --epochs 100
    * $ python tools/train_basicnet.py --network minivggnet --model output/cifar10_minivggnet_without_dropout.hdf5 --epochs 200
    
2) Train deep neural networks:

    * $ python tools/train_deepnet.py --network alexnet --model output/cifar10_alexnet.hdf5 --dropout 1 --batch-size 50 --epochs 35
    * $ python tools/train_deepnet.py --network vgg16 --model output/cifar10_vgg16.hdf5 --dropout 1 --batch-size 50 --epochs 35

## Our results/observations on CIFAR-10:

        Network    |    Train (loss)    |    Train (accuracy)    |    Test (accuracy)
    ---------------|--------------------|------------------------|---------------------
       ShallowNet  |       0.5428       |        0.8117          |       0.5387
    ---------------|--------------------|------------------------|---------------------
         LeNet     |       0.9753       |         1.0            |       0.7065
    ---------------|--------------------|------------------------|---------------------
      KarpathyNet  |       1.0452       |        0.6323          |       0.6715
    ---------------|--------------------|------------------------|---------------------
        miniVGG    |       0.5045       |        0.8411          |       0.75
    ---------------|--------------------|------------------------|---------------------
        AlexNet    |       0.0011       |        0.9999          |       0.7661
        
## Contact

Please feel free to leave suggestions or comments to Franck FOTSO (romyny9096@gmail.com)

    
    
