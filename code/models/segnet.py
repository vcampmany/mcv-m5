from keras.models import Model
from keras import models
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
from keras.regularizers import l2
import keras.backend as K

# Custom layers import
from layers.ourlayers import NdSoftmax, CropLayer2D, DePool2D

def channel_idx():
	if K.image_dim_ordering() == 'tf':
		channel = 3
	else:
		channel = 1

	return channel

# functions from github https://github.com/dvazquezcvc/mcv-m5/issues/32
def downsampling_block_basic(inputs, n_filters, filter_size,W_regularizer=None):
	# This extra padding is used to prevent problems with different input
	# sizes. At the end the crop layer remove extra paddings
	pad = ZeroPadding2D(padding=(1, 1))(inputs)
	conv = Convolution2D(n_filters, filter_size, filter_size, border_mode='same', W_regularizer=W_regularizer)(pad)
	bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
	act = Activation('relu')(bn)
	maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act)
	return maxp

def upsampling_block_basic(inputs, n_filters, filter_size, unpool_layer=None, W_regularizer=None, use_unpool=True):
	if use_unpool:
		up = DePool2D(unpool_layer)(inputs)
	else:
		up = UpSampling2D()(inputs)
	conv = Convolution2D(n_filters, filter_size, filter_size, border_mode='same', W_regularizer=W_regularizer)(up)
	bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
	return bn


# based on https://github.com/imlab-uiip/keras-segnet/blob/master/build_model.py
def downsampling_block_vgg(inputs, n_filters, filter_size, n_convs, W_regularizer=None):
	# This extra padding is used to prevent problems with different input
	# sizes. At the end the crop layer remove extra paddings
	conv = ZeroPadding2D(padding=(1, 1))(inputs)
	
	for i in range(n_convs): # in VGG there are several 3x3 conv layers together
		conv = Convolution2D(n_filters, filter_size, filter_size, border_mode='same', W_regularizer=W_regularizer)(conv)
		bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
		conv = Activation('relu')(bn)

	maxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
	return maxp

def upsampling_block_vgg(inputs, n_filters, filter_size, n_convs, unpool_layer=None, W_regularizer=None, use_unpool=True):
	# there is only one upsampling at the beginning
	if use_unpool:
		up = DePool2D(unpool_layer)(inputs)
	else:
		up = UpSampling2D()(inputs)

	conv = up
	for i in range(n_convs): # in VGG there are several 3x3 conv layers together
		conv = Convolution2D(n_filters, filter_size, filter_size, border_mode='same', W_regularizer=W_regularizer)(conv)
		bn = BatchNormalization(mode=0, axis=channel_idx())(conv)
		conv = Activation('relu')(bn)

	return conv

def build_segnet_basic(inputs, n_classes, depths=[64, 64, 64, 64], filter_size=7, l2_reg=0.):
	""" encoding layers """
	enc1 = downsampling_block_basic(inputs, depths[0], filter_size, l2(l2_reg))
	enc2 = downsampling_block_basic(enc1, depths[1], filter_size, l2(l2_reg))
	enc3 = downsampling_block_basic(enc2, depths[2], filter_size, l2(l2_reg))
	enc4 = downsampling_block_basic(enc3, depths[3], filter_size, l2(l2_reg))

	""" decoding layers """
	dec1 = upsampling_block_basic(enc4, depths[3], filter_size, enc4, l2(l2_reg))
	dec2 = upsampling_block_basic(dec1, depths[2], filter_size, enc3, l2(l2_reg))
	dec3 = upsampling_block_basic(dec2, depths[1], filter_size, enc2, l2(l2_reg))
	dec4 = upsampling_block_basic(dec3, depths[0], filter_size, enc1, l2(l2_reg))

	""" logits """
	l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec4)
	score = CropLayer2D(inputs, name='score')(l1)
	softmax_segnet = NdSoftmax()(score)

	# Complete model
	model = Model(input=inputs, output=softmax_segnet)

	return model

def build_segnet_vgg(inputs, n_classes, depths=[64, 128, 256, 512, 512], filter_size=3, l2_reg=0.):
	""" encoding layers """
	enc1 = downsampling_block_vgg(inputs, depths[0], filter_size, 2, l2(l2_reg))
	enc2 = downsampling_block_vgg(enc1, depths[1], filter_size, 2, l2(l2_reg))
	enc3 = downsampling_block_vgg(enc2, depths[2], filter_size, 3, l2(l2_reg))
	enc4 = downsampling_block_vgg(enc3, depths[3], filter_size, 3, l2(l2_reg))
	enc5 = downsampling_block_vgg(enc4, depths[4], filter_size, 3, l2(l2_reg))

	""" decoding layers """
	dec1 = upsampling_block_vgg(enc4, depths[4], filter_size, 3, enc5, l2(l2_reg))
	dec2 = upsampling_block_vgg(dec1, depths[3], filter_size, 3, enc4, l2(l2_reg))
	dec3 = upsampling_block_vgg(dec2, depths[2], filter_size, 3, enc3, l2(l2_reg))
	dec4 = upsampling_block_vgg(dec3, depths[1], filter_size, 2, enc2, l2(l2_reg))
	dec5 = upsampling_block_vgg(dec3, depths[0], filter_size, 2, enc1, l2(l2_reg))

	""" logits """
	l1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(dec5)
	score = CropLayer2D(inputs, name='score')(l1)
	softmax_segnet = NdSoftmax()(score)

	# Complete model
	model = Model(input=inputs, output=softmax_segnet)

	return model

def build_segnet(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0., freeze_layers_from='base_model', 
				path_weights=None, basic=False):

	inputs = Input(img_shape)
	if basic:
		model = build_segnet_basic(inputs, n_classes)
	else:
		model = build_segnet_vgg(inputs, n_classes)

	return model


