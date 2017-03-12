# Implemented by: Guillem Cucurull, gcucurull@gmail.com, (https://github.com/gcucurull)
# Based on:
# https://github.com/szagoruyko/wide-residual-networks/tree/master/pretrained
# https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/wide-resnet.lua
# https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb

# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, BatchNormalization, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D, AveragePooling2D)

#from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape
import keras.backend as K

import urllib
import os

# transpose weights if loading from pre-trained weights file
def transpose(v):
    if v.ndim == 4:
        return v.transpose(2,3,1,0)
    elif v.ndim == 2:
        return v.transpose()
    return v

def residual_group(x, base, stride, blocks, bn_axis, n_filters):
    for i in range(blocks):
        b_base = ('%s.block%d.conv') % (base, i)
        in_ = x
        x = Convolution2D(n_filters[0], 1, 1, name=b_base+'0')(in_) # 128 should be variable
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D((1, 1))(x) # add padding
        sub = i==0 and stride or 1
        x = Convolution2D(n_filters[1], 3, 3, subsample=(sub, sub), name=b_base+'1')(x) # 128 should be variable
        x = BatchNormalization(axis=bn_axis)(x)
        x = Activation('relu')(x)

        x = Convolution2D(n_filters[2], 1, 1, name=b_base+'2')(x) # 256 should be variable
        x = BatchNormalization(axis=bn_axis)(x)

        if i == 0:
            shortcut = Convolution2D(n_filters[2], 1, 1, name=b_base+'_dim', subsample=(stride, stride))(in_) # 256 should be variable
            shortcut = BatchNormalization(axis=bn_axis)(shortcut)
        else:
            shortcut = in_
        x = merge([x, shortcut], mode='sum')
        x = Activation('relu')(x)
    return x

def build_wideresnet(img_shape=(3, 224, 224), n_classes=1000, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model'):
    # Decide if load pretrained weights from imagenet

    if load_pretrained:
        import hickle as hkl
        weights = 'imagenet'
        # download weights if needed
        if not os.path.isfile("wide-resnet-50-2-export.hkl"):
            print('   Downloading weights')
            urllib.urlretrieve ("https://s3.amazonaws.com/pytorch/h5models/wide-resnet-50-2-export.hkl", "wide-resnet-50-2-export.hkl")
        
        imagenet_weights = hkl.load('wide-resnet-50-2-export.hkl')
    else:
        weights = None
        
    # Determine proper input shape
    input_shape = _obtain_input_shape(img_shape,
                                      default_size=224,
                                      min_size=197,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=False)

    img_input = Input(shape=input_shape)

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # conv 0
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv0')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_0')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # group 1
    x = residual_group(x, 'group0', 1, 3, bn_axis, [128, 128, 256])
    # group 2
    x = residual_group(x, 'group1', 2, 4, bn_axis, [256, 256, 512])
    # group 3
    x = residual_group(x, 'group2', 2, 6, bn_axis, [512, 512, 1024])
    # group 4
    x = residual_group(x, 'group3', 2, 3, bn_axis, [1024, 1024, 2048])

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)

    base_model = Model(img_input, x)

    # output layer
    predictions = Dense(n_classes, activation='softmax', name='output')(x)
    model = Model(input=base_model.input, output=predictions, name='wideresnet')

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    # load pre-trained weights
    if load_pretrained:
        print('   Loading pre-trained weights...')
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights() # list of numpy arrays
            name = layer.name

            if name+'.weight' in imagenet_weights.keys() and name+'.bias' in imagenet_weights.keys():
                weights = transpose(imagenet_weights[name+'.weight'])
                bias = transpose(imagenet_weights[name+'.bias'])

                assert weights.shape == layer.get_weights()[0].shape
                assert bias.shape == layer.get_weights()[1].shape

                layer.set_weights([weights, bias])

    return model
