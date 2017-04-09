from keras.models import Model
from keras import models
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot

# Custom layers import
from layers.ourlayers import NdSoftmax, CropLayer2D, DePool2D


def build_segnet(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0., freeze_layers_from='base_model', 
				path_weights=None, basic=False):

	kernel = 3
	block1_filter = 64
	block2_filter = 128
	block3_filter = 256
	block4_filter = 512
	block5_filter = 512

    # VGG like SegNet
	if not basic:
		#img_shape=(224, 224, 3)
		inputs = Input(img_shape)
		###########################################3
		## ENCODING BLOCK
		###########################################3
		# VGG block 1
		conv1_1 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(inputs)
		bnorm1_1 = BatchNormalization()(conv1_1)
		act1_1 = Activation('relu')(bnorm1_1)
		conv1_2 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(act1_1)
		bnorm1_2 = BatchNormalization()(conv1_2)
		act1_2 = Activation('relu')(bnorm1_2)
		pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act1_2)

		# VGG block 2
		conv2_1 = Convolution2D(block2_filter, kernel, kernel, border_mode='same')(pool1_1)
		bnorm2_1 = BatchNormalization()(conv2_1)
		act2_1 = Activation('relu')(bnorm2_1)
		conv2_2 = Convolution2D(block2_filter, kernel, kernel, border_mode='same')(act2_1)
		bnorm2_2 = BatchNormalization()(conv2_2)
		act2_2 = Activation('relu')(bnorm2_2)
		pool2_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act2_2)

		# VGG block 3
		conv3_1 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(pool2_1)
		bnorm3_1 = BatchNormalization()(conv3_1)
		act3_1 = Activation('relu')(bnorm3_1)
		conv3_2 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act3_1)
		bnorm3_2 = BatchNormalization()(conv3_2)
		act3_2 = Activation('relu')(bnorm3_2)
		conv3_3 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act3_2)
		bnorm3_3 = BatchNormalization()(conv3_3)
		act3_3 = Activation('relu')(bnorm3_3)
		pool3_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act3_3)

		# VGG block 4
		conv4_1 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(pool3_1)
		bnorm4_1 = BatchNormalization()(conv4_1)
		act4_1 = Activation('relu')(bnorm4_1)
		conv4_2 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act4_1)
		bnorm4_2 = BatchNormalization()(conv4_2)
		act4_2 = Activation('relu')(bnorm4_2)
		conv4_3 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act4_2)
		bnorm4_3 = BatchNormalization()(conv4_3)
		act4_3 = Activation('relu')(bnorm4_3)
		pool4_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act4_3)

		# VGG block 5
		conv5_1 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(pool4_1)
		bnorm5_1 = BatchNormalization()(conv5_1)
		act5_1 = Activation('relu')(bnorm5_1)
		conv5_2 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act5_1)
		bnorm5_2 = BatchNormalization()(conv5_2)
		act5_2 = Activation('relu')(bnorm5_2)
		conv5_3 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act5_2)
		bnorm5_3 = BatchNormalization()(conv5_3)
		act5_3 = Activation('relu')(bnorm5_3)
		pool5_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act5_3)

		###########################################3
		## DECODING BLOCK
		###########################################3
		# Decoder block 1
		up1_1 = UpSampling2D(size=(2, 2))(pool5_1)
		#up1_1 = DePool2D(pool5_1, size=(2, 2), dim_ordering='tf')(pool5_1)
		conv6_1 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(up1_1)
		bnorm6_1 = BatchNormalization()(conv6_1)
		act6_1 = Activation('relu')(bnorm6_1)
		conv6_2 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act6_1)
		bnorm6_2 = BatchNormalization()(conv6_2)
		act6_2 = Activation('relu')(bnorm6_2)
		conv6_3 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act6_2)
		bnorm6_3 = BatchNormalization()(conv6_3)
		act6_3 = Activation('relu')(bnorm6_3)

		# Decoder block 2
		up2_1 = UpSampling2D(size=(2, 2))(act6_3)
		#up2_1 = DePool2D(pool4_1, size=(2,2), dim_ordering='tf')(act1_3)
		conv7_1 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(up2_1)
		bnorm7_1 = BatchNormalization()(conv7_1)
		act7_1 = Activation('relu')(bnorm7_1)
		conv7_2 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act7_1)
		bnorm7_2 = BatchNormalization()(conv7_2)
		act7_2 = Activation('relu')(bnorm7_2)
		conv7_3 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act7_2)
		bnorm7_3 = BatchNormalization()(conv7_3)
		act7_3 = Activation('relu')(bnorm7_3)

		# Decoder block 3
		up3_1 = UpSampling2D(size=(2, 2))(act7_3)
		#up3_1 = DePool2D(pool3_1, size=(2,2), dim_ordering='tf')(act2_3)
		conv8_1 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(up3_1)
		bnorm8_1 = BatchNormalization()(conv8_1)
		act8_1 =Activation('relu')(bnorm8_1)
		conv8_2 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act8_1)
		bnorm8_2 = BatchNormalization()(conv8_2)
		act8_2 = Activation('relu')(bnorm8_2)
		conv8_3 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act8_2)
		bnorm8_3 = BatchNormalization()(conv8_3)
		act8_3 = Activation('relu')(bnorm8_3)

		# Decoder block 4
		up4_1 = UpSampling2D(size=(2, 2))(act8_3)
		#up4_1 = DePool2D(pool3_1, size=(2, 2), dim_ordering='tf')(act3_3)
		conv9_1 = Convolution2D(block2_filter, kernel, kernel, border_mode='same')(up4_1)
		bnorm9_1= BatchNormalization()(conv9_1)
		act9_1 = Activation('relu')(bnorm9_1)
		conv9_2 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act9_1)
		bnorm9_2 = BatchNormalization()(conv9_2)
		act9_2 = Activation('relu')(bnorm9_2)

		# Decoder block 5
		up5_1 = UpSampling2D(size=(2, 2))(act9_2)
		#up5_1 = DePool2D(pool2_1, size=(2, 2), dim_ordering='tf')(act4_2)
		conv10_1 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(up5_1)
		bnorm10_1 = BatchNormalization()(conv10_1)
		act10_1 = Activation('relu')(bnorm10_1)
		conv10_2 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(act10_1)
		bnorm10_2 = BatchNormalization()(conv10_2)
		#act5_2 = Activation('relu')(bnorm5_2)
		
		# Fit channels to number of classes 
		conv11_1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(bnorm10_2)
		
		softmax = NdSoftmax()(conv11_1)
#		print softmax.shape
#		quit()
		model = Model(input=inputs, output=softmax)

	else:
		print "basic segnet not implemented yet"

	return model


