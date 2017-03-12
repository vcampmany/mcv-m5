# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)

from keras_vgg16_l2reg import VGG16_l2reg
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.regularizers import l2
# Paper: https://arxiv.org/pdf/1409.1556.pdf

def build_vgg(img_shape=(3, 224, 224), n_classes=1000, n_layers=16, l2_reg=0.,
                load_pretrained=False, freeze_layers_from='base_model', out_name=None):
                
    # Decide if load pretrained weights from imagenet
    if load_pretrained:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model
    if n_layers==16:
    	# Call vgg model with l2 regularization
        base_model = VGG16_l2reg(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape,
                           l2_regu=l2_reg)
    elif n_layers==19:
        base_model = VGG19(include_top=False, weights=weights,
                           input_tensor=None, input_shape=img_shape)
    else:
        raise ValueError('Number of layers should be 16 or 19')
        
    # Add final layers
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(4096, activation='relu', name='dense_1', W_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='dense_2', W_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)
    last_name = 'dense_3'
    if out_name:
        last_name += '_'+out_name+'_'+str(n_classes)
    x = Dense(n_classes, name=last_name, W_regularizer=l2(l2_reg))(x)
    predictions = Activation("softmax", name="softmax")(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)
		
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

    return model
