from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,concatenate
from tensorflow.keras.layers import Dense,Reshape,Flatten,Activation,Concatenate
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.losses import Huber

# define an encoder block
def define_encoder_block(layer_in, n_filters, batchnorm=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add downsampling layer
	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# conditionally add batch normalization
	if batchnorm:
		g = BatchNormalization()(g, training=True)
	# leaky relu activation
	g = LeakyReLU(alpha=0.2)(g)
	return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# add upsampling layer
	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
	# add batch normalization
	g = BatchNormalization()(g, training=True)
	# conditionally add dropout
	if dropout:
		g = Dropout(0.5)(g, training=True)
	# merge with skip connection
	g = Concatenate()([g, skip_in])
	# relu activation
	g = Activation('relu')(g)
	return g

# define the standalone generator model
def define_generator(image_shape = (64,512,1)):
	# weight initialization
	init = RandomNormal(stddev=0.02)

	# Image input
	in_image = Input(shape=image_shape)

	# Encoder model with increased filters
	e1 = define_encoder_block(in_image, 256, batchnorm=False) 
	e2 = define_encoder_block(e1, 512)
	e3 = define_encoder_block(e2, 512) 
	e4 = define_encoder_block(e3, 1024) 
	e5 = define_encoder_block(e4, 1024) 

	# Bottleneck with residual connection
	#b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
	b = Conv2D(1024, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
	b = Activation('relu')(b)


	# Decoder model with increased complexity and residual connections
	d1 = decoder_block(b, e5, 1024)
	d2 = decoder_block(d1, e4, 1024)  
	d3 = decoder_block(d2, e3, 512)
	d4 = decoder_block(d3, e2, 512, dropout=False)
	d5 = decoder_block(d4, e1, 256, dropout=False)
	# Output
	g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
	out_image = Activation('sigmoid')(g)
	# Define model
	model = Model(in_image, out_image)
	delta = 1.0
	huber_loss = Huber(delta=delta)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	model.compile(loss=huber_loss, optimizer=opt, loss_weights=[100])
	return model

