# --------------------------------------------------------
# Written by Romyny
# Based on 'heuritech' github repository:
# https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/customlayers.py
# License MIT
# --------------------------------------------------------
from keras.layers.core import  Lambda

def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = int(X.shape[axis] // ratio_split)
        #print ('X.shape: {}'.format(X.shape))
        #print ('id_split: {}'.format(id_split))
        #print ('div: {}'.format(div))

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output = X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)
