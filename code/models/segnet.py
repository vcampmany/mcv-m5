from keras.models import Model
from keras import models
from keras.layers import Input
from keras.layers.core import Activation, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot

# Custom layers import
from layers.ourlayers import NdSoftmax, CropLayer2D

def create_encoding_layers(input_shape):
	kernel = 3
	block1_filter = 64
	block2_filter = 128
	block3_filter = 256
	block4_filter = 512
	block5_filter = 512
		
	return [
			# VGG block 1
			Convolution2D(block1_filter, kernel, kernel, border_mode='same', input_shape=input_shape),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block1_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG block 2
			Convolution2D(block2_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block2_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG block 3
			Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG block 4
			Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

			# VGG block 5
			Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
			BatchNormalization(),
			Activation('relu'),
			MaxPooling2D(pool_size=(2, 2), strides=(2, 2))	
		]

def create_decoding_layers():
	kernel = 3
	block1_filter = 64
	block2_filter = 128
	block3_filter = 256
	block4_filter = 512
	block5_filter = 512

	return [
		# Decoder block 1
		UpSampling2D(size=(2, 2)),
		Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block5_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),

		# Decoder block 2
		UpSampling2D(size=(2, 2)),
		Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block4_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),

		# Decoder block 3
		UpSampling2D(size=(2, 2)),
		Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),

		# Decoder block 4
		UpSampling2D(size=(2, 2)),
		Convolution2D(block2_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block3_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),

		# Decoder block 5
		UpSampling2D(size=(2, 2)),
		Convolution2D(block1_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
		Convolution2D(block1_filter, kernel, kernel, border_mode='same'),
		BatchNormalization(),
		Activation('relu'),
	]


def build_segnet(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0., freeze_layers_from='base_model', 
				path_weights=None, basic=False):

	kernel = 3
	block1_filter = 64
	block2_filter = 128
	block3_filter = 256
	block4_filter = 512
	block5_filter = 512
	#img_shape=(224, 224, 3)

    # VGG like SegNet
	if not basic:
		#autoencoder = models.Sequential()

		#autoencoder.encoding_layers = create_encoding_layers(img_shape)
		#for l in autoencoder.encoding_layers:
		#	autoencoder.add(l)

		#autoencoder.decoding_layers = create_decoding_layers()
		#for l in autoencoder.decoding_layers:
		#	autoencoder.add(l)

		#autoencoder.add(Convolution2D(n_classes+1, 1, 1, border_mode='valid',))
		#plot(autoencoder, to_file='model.png', show_shapes=True)
		#data_shape = 224*224
		#autoencoder.add(Reshape((n_classes+1, data_shape), input_shape=(224, 224, n_classes+1)))
		#autoencoder.add(Permute((2, 1)))
		#autoencoder.add(Activation('softmax'))
		# Softmax
		#autoencoder.add(CropLayer2D(img_shape))
		#autoencoder.add(NdSoftmax())



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
		bnorm3_2 = BatchNormalization()(conv3_1)
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
		up1_1 = UpSampling2D(size=(2, 2)),(pool5_1)
		conv1_1 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(pool5_1)
		bnorm1_1 = BatchNormalization()(conv1_1)
		act1_1 = Activation('relu')(bnorm1_1)
		conv1_2 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act1_1)
		bnorm1_2 = BatchNormalization()(conv1_2)
		act1_2 = Activation('relu')(bnorm1_2)
		conv1_3 = Convolution2D(block5_filter, kernel, kernel, border_mode='same')(act1_2)
		bnorm1_3 = BatchNormalization()(conv1_3)
		act1_1 = Activation('relu')(bnorm1_3)

		# Decoder block 2
		up2_1 = UpSampling2D(size=(2, 2))(act1_1)
		conv2_1 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(up2_1)
		bnorm2_1 = BatchNormalization()(conv2_1)
		act2_1 = Activation('relu')(bnorm2_1)
		conv2_2 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act2_1)
		bnorm2_2 = BatchNormalization()(conv2_2)
		act2_2 = Activation('relu')(bnorm2_2)
		conv2_3 = Convolution2D(block4_filter, kernel, kernel, border_mode='same')(act2_2)
		bnorm2_3 = BatchNormalization()(conv2_3)
		act2_3 = Activation('relu')(bnorm2_3)

		# Decoder block 3
		up3_1 = UpSampling2D(size=(2, 2))(act2_3)
		conv3_1 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(up3_1)
		bnorm3_1 = BatchNormalization()(conv3_1)
		act3_1 =Activation('relu')(bnorm3_1)
		conv3_2 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act3_1)
		bnorm3_2 = BatchNormalization()(conv3_2)
		act3_2 = Activation('relu')(bnorm3_2)
		conv3_3 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act3_2)
		bnorm3_3 = BatchNormalization()(conv3_3)
		act3_3 = Activation('relu')(bnorm3_3)

		# Decoder block 4
		up4_1 = UpSampling2D(size=(2, 2))(act3_3)
		conv4_1 = Convolution2D(block2_filter, kernel, kernel, border_mode='same')(up4_1)
		bnorm4_1= BatchNormalization()(conv4_1)
		act4_1 = Activation('relu')(bnorm4_1)
		conv4_2 = Convolution2D(block3_filter, kernel, kernel, border_mode='same')(act4_1)
		bnorm4_2 = BatchNormalization()(conv4_2)
		act4_2 = Activation('relu')(bnorm4_2)

		# Decoder block 5
		up5_1 = UpSampling2D(size=(2, 2))(act4_2)
		conv5_1 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(up5_1)
		bnorm5_1 = BatchNormalization()(conv5_1)
		act5_1 = Activation('relu')(bnorm5_1)
		conv5_2 = Convolution2D(block1_filter, kernel, kernel, border_mode='same')(act5_1)
		bnorm5_2 = BatchNormalization()(conv5_2)
		#act5_2 = Activation('relu')(bnorm5_2)

		# Fit channels to number of classes
		conv7_1 = Convolution2D(n_classes, 1, 1, border_mode='valid')(bnorm5_2)

		softmax = NdSoftmax()(score)

		model = Model(input=inputs, output=softmax)

	else:
		print "basic segnet not implemented yet"

	return model


