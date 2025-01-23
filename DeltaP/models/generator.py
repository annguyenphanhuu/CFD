from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Dropout,BatchNormalization
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from keras.layers import LeakyReLU


# define the standalone generator model
def define_cnn_model(image_shape=(64, 512, 1)):
	# Khởi tạo RandomNormal với stddev = 0.02
	init = RandomNormal(stddev=0.02)

	# Định nghĩa đầu vào
	inputs = Input(shape=image_shape)

	# Convolutional layers
	g1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', kernel_initializer=init)(inputs)
	g1 = LeakyReLU(alpha=0.2)(g1)

	g2 = MaxPooling2D(pool_size=(2, 2))(g1)

	g3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=init)(g2)
	g3 = LeakyReLU(alpha=0.2)(g3)

	g4 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=init)(g3)
	g5 = MaxPooling2D(pool_size=(2, 2))(g4)

	g6 = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer=init)(g5)
	g7 = MaxPooling2D(pool_size=(2, 2))(g6)

	# Flatten layer
	flatten = Flatten()(g7)

	# Fully connected layers
	fc1 = Dense(256, activation='relu', kernel_initializer=init)(flatten)
	dropout = Dropout(0.5)(fc1)
	fc2 = Dense(128, activation='relu', kernel_initializer=init)(dropout)

	# Output layer
	outputs = Dense(1, activation='linear', kernel_initializer=init)(fc2)

	# Tạo mô hình
	model = Model(inputs=inputs, outputs=outputs)

	optimizer = Adam(learning_rate=0.001, beta_1=0.5)
	# Compile the model
	model.compile(optimizer=optimizer, loss=Huber(delta=1.0), loss_weights = [100])

	return model

