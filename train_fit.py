from ftplib import error_perm
from xmlrpc.client import Boolean
from numpy import *
from pandas import *
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import csv
import cv2
import time
import matplotlib.pyplot as plt
#matplotlib.use('agg')
import skfmm
import random
from random import choice
from keras import backend as K
import gc
import psutil
from scipy.interpolate import griddata
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras import Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,LeakyReLU,concatenate
from tensorflow.keras.layers import Dense,Reshape,Flatten,Activation,Concatenate
from tensorflow.keras.layers import Dropout,BatchNormalization
import argparse

parser = argparse.ArgumentParser(description='Input option for training GAN CFD model')
parser.add_argument("-c", "--checkpoint", default=True, help="Whether load checkpoint for training", type=Boolean)
parser.add_argument("-e", "--epocheval", default=5, help="Number of epoch for each evaluation", type=int)
parser.add_argument("-n", "--numepoch", default=10000, help="Number of epoch for training iteration", type=int)
parser.add_argument("-d", "--directory", default='training_GAN_CFD/', help="training directory output", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_src_image = Input(shape=image_shape)
	# target image input
	in_target_image = Input(shape=image_shape)
	# concatenate images channel-wise
	merged = Concatenate()([in_src_image, in_target_image])
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = BatchNormalization()(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output as 16x16 PatchGAN model
	d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	patch_out = Activation('sigmoid')(d)
	# define model
	model = Model([in_src_image, in_target_image], patch_out)
	# compile model
	opt = Adam(learning_rate=0.0002, beta_1=0.5)
	# slow weight update to 0.5 in respect to generator
	model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
	return model

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
def define_generator(image_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # Image input
    in_image = Input(shape=image_shape)
    
    # Encoder model with increased filters
    e1 = define_encoder_block(in_image, 128, batchnorm=False)  # (None, 32, 256, 128)
    e2 = define_encoder_block(e1, 256)  # (None, 16, 128, 256)
    e3 = define_encoder_block(e2, 512)  # (None, 8, 64, 512)
    e4 = define_encoder_block(e3, 1024)  # (None, 4, 32, 1024)
    e5 = define_encoder_block(e4, 1024)  # (None, 2, 16, 1024) 

    # Bottleneck with residual connection
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
	
    # Decoder model with increased complexity and residual connections
    d1 = decoder_block(b, e5, 1024)
    d2 = decoder_block(d1, e4, 1024)  
    d3 = decoder_block(d2, e3, 512)
    d4 = decoder_block(d3, e2, 512, dropout=False)
    d5 = decoder_block(d4, e1, 256, dropout=False)

    # Output layer with increased complexity
    g = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d5)
    out_image = Activation('tanh')(g)
    
    # Define model
    model = Model(in_image, out_image)
    
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model, image_shape):
	# make weights in the discriminator not trainable
	for layer in d_model.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# define the source image
	in_src = Input(shape=image_shape)
	# connect the source image to the generator input
	gen_out = g_model(in_src)
	# connect the source input and generator output to the discriminator input
	dis_out = d_model([in_src, gen_out])
	# src image as input, generated image and classification output
	model = Model(in_src, [dis_out, gen_out])
	# compile model
	opt = Adam(learning_rate=0.0004, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

# select a batch of random samples, returns images and target
def generate_real_samples(x_train_1, y_train_2, ix, patch_shape):
	# retrieve selected images
	X1 = []
	X2 = []
	for i in ix:
		x1 = x_train_1[i]
		X1.append(x1)
		x2 = y_train_2[i]
		X2.append(x2)
	# generate 'real' class labels (1)
	y = np.ones((len(ix), patch_shape, patch_shape * 8, 1))
	X1 = np.asarray(X1).astype('float32')
	X2 = np.asarray(X2).astype('float32')	
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, X_realA, patch_shape):
	# generate fake instance
	X = []
	for i in range(len(X_realA)):
		x = g_model.predict(X_realA[[i]])
		x = np.asarray(x[0])
		X.append(x)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape * 8, 1))
	X = np.asarray(X).astype('float32')
	return X, y

def load_data(data_file, nx, ny, nx_new, ny_new):
    try:
        # Đọc dữ liệu từ CSV
        data = pd.read_csv(data_file)
        
        # Kiểm tra các cột cần thiết
        required_columns = ['y_center', 'z_center', 'p_center', 'Umean_center']
        if not all(col in data.columns for col in required_columns):
            print("Missing required columns in CSV file!")
            return []
        
        # Lấy dữ liệu từ các cột
        xlist = data['y_center'].values
        ylist = data['z_center'].values
        temp = data['p_center'].values
        vx = data['Umean_center'].values
        
        # Xử lý dữ liệu
        data_size = nx * ny
        if len(xlist) == data_size:
            if nx == nx_new or nx_new is None:
                xcor = np.ones((nx, ny))
                ycor = np.ones((nx, ny))
                P_array = np.ones((nx, ny))
                U_array = np.ones((nx, ny))
                for i in range(nx):
                    for j in range(ny):
                        xcor[i, j] = xlist[(j + nx * i)]
                        ycor[i, j] = ylist[(j + nx * i)]
                        P_array[i, j] = temp[(j + nx * i)]
                        U_array[i, j] = vx[(j + nx * i)]
            else:
                # Nội suy nếu kích thước cần thay đổi
                XN, YN = np.meshgrid(
                    np.linspace(min(xlist), max(xlist), nx_new),
                    np.linspace(min(ylist), max(ylist), ny_new),
                )
                P_array = griddata((xlist, ylist), temp, (XN, YN), method='nearest')
                U_array = griddata((xlist, ylist), vx, (XN, YN), method='nearest')
                xcor = XN
                ycor = YN
            return xcor, ycor, P_array, U_array
        else:
            print("Missing data file!")
            return []
    except ValueError as e:
        print(f"Error processing data file: {e}")
        return []

def plot_image(var, pretext, fieldname, flag):
	"""
	Generates and saves images for the given data array.
	"""
	if flag == 1:
		labeltxt = 'SDF Boundary'
	elif flag == 2:
		labeltxt = 'Pressure (Pa)'
	elif flag == 3:
		labeltxt = 'U mean (m/s)'

	Z, Y = np.meshgrid(np.linspace(0, 50, 512), np.linspace(0, 4, 64))
	fig = plt.figure()	
	ax = fig.gca()

	plt.gca().set_aspect('equal', adjustable='box')
	plt.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow)
	plt.colorbar(label=labeltxt)

	# Save the plot to the specific folder for this CSV file
	plt.savefig(args.directory+pretext+fieldname+'.png')
	fig.clear() 
	ax.clear() 
	plt.close(fig)
	del fig
    
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, x_test_1, y_test_2, test_size, field):
	# generate a batch of fake samples
	ix = random.randint(0, test_size-1)
	X_fakeB, _ = generate_fake_samples(g_model, x_test_1[[ix]], 1)
	
	X = x_test_1[ix]
	X = np.squeeze(X, axis=2)
	y = X_fakeB[0]
	y = np.squeeze(y, axis=2)
	Y = y_test_2[ix]
	Y = np.squeeze(Y, axis=2)
	print('X shape',X.shape)
	print('Y shape',Y.shape)
	print('y shape',y.shape)

	plot_image(X,str(step+1)+'_Boundary_',field,1)
	plot_image(Y,str(step+1)+'_CFD_',field,3)
	plot_image(y,str(step+1)+'_Predict_',field,3)
	
	y_error = abs(Y-y)
	plot_image(y_error,str(step+1)+'_error_abs_',field,3)

	del X,Y,y,y_error,X_fakeB,_

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

# train pix2pix models
def train(d_model, g_model, gan_model, n_epochs=args.numepoch, batch_size=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	bnd_array = []
	sflow_P_array = []
	sflow_U_array = []

	processed_folder = "./output_folder/processed/"
	for file_path in os.listdir(processed_folder):
		with open(processed_folder + file_path, 'r') as filename:
			file = csv.DictReader(filename)
			P_array = []
			U_array = []
			pts = np.ones((40,500))
			p = np.zeros((40,500))
			u = np.zeros((40,500))

			# Extract data from CSV columns
			for col in file:
				P_array.append(float(col['p_center']))
				U_array.append(float(col['Umean_center']))
			# Map data into the domain (4x50 mm)
			for i in range(40):
				for j in range(500):
					p[i, j] = P_array[j + i * 500]
					u[i, j] = U_array[j + i * 500]
					if P_array[j + i * 500] == 0:
						pts[i, j] = 1
					else:
						pts[i, j] = -1

			# Compute SDF (Signed Distance Function)
			pts = cv2.resize(pts, (512, 64), interpolation=cv2.INTER_NEAREST)
			phi = pts
			d_x = 4 / 64
			d = skfmm.distance(phi, d_x)

			bnd = np.array(d)
			bnd_array.append(bnd)

		if os.path.isfile(processed_folder + file_path):
			print("processing mean file:", file_path)
			data_list = load_data(processed_folder + file_path,500,40,512,64)
			if len(data_list) > 0:
		# get data list in field
				[y_center,z_center,p_center,Umean_center] = data_list
				P_array = np.asarray(p_center)
				U_array = np.asarray(Umean_center)
				sflow_P_array.append(P_array)
				sflow_U_array.append(U_array)
			else:
				print('error data list!')
		
	

	x_train_1 = bnd_array[:1000]
	x_test_1 = bnd_array[-190:]
	# y_train_1 = sflow_P_array[:1000]
	# y_test_1 = sflow_P_array[-190:]
	y_train_2 = sflow_U_array[:1000]
	y_test_2 = sflow_U_array[-190:]
	data_size = len(x_train_1)
	test_size = len(x_test_1)

	x_train_1 = np.asarray(x_train_1).astype('float32')
	# y_train_1 = np.asarray(y_train_1).astype('float32')
	y_train_2 = np.asarray(y_train_2).astype('float32')
	x_train_1 = np.expand_dims(x_train_1,axis=3)
	# y_train_1 = np.expand_dims(y_train_1,axis=3)
	y_train_2 = np.expand_dims(y_train_2,axis=3)

	# normalize data to range [0,1]
	# U_max = y_train_2.max()
	# U_min = y_train_2.min()
	# y_train_2 = (y_train_2 - U_min) / (U_max - U_min)
	# y_train_1 = (y_train_1 - 0.5) / 0.5

	x_test_1 = np.asarray(x_test_1).astype('float32')
	# y_test_1 = np.asarray(y_test_1).astype('float32')
	y_test_2 = np.asarray(y_test_2).astype('float32')
	x_test_1 = np.expand_dims(x_test_1,axis=3)
	# y_test_1 = np.expand_dims(y_test_1,axis=3)
	y_test_2 = np.expand_dims(y_test_2,axis=3)
	# y_test_2 = (y_test_2 - U_min) / (U_max - U_min)
	# y_test_1 = (y_test_1 - 0.5) / 0.5
	
	print('x1 shape',x_train_1.shape)
	# print('y1 shape',y_train_1.shape,y_train_1.min(),y_train_1.max())
	print('y2 shape',y_train_2.shape,y_train_2.min(),y_train_2.max())
	print('x1 test shape',x_test_1.shape)
	# print('y1 test shape',y_test_1.shape,y_test_1.min(),y_test_1.max())
	print('y2 test shape',y_test_2.shape,y_test_2.min(),y_test_2.max())
 
	# Define the number of batches per epoch
	bat_per_epo = int(data_size / batch_size)
	# Number of training iterations
	n_steps = bat_per_epo * n_epochs
	idx_choice = [i for i in range(data_size)]
	d_1 = d_2 = g_ = 0
	count_epoch = 0
	total_d1 = []
	total_d2 = []
	total_g  = []
	total_val = []
	min_eval = -1

	init_t = time.time()

	

            
# LOAD TRAINING CHECKPOINTS

image_shape = (64,512,1)
chkpt = args.checkpoint

if chkpt == True:
	d_model = load_model('./training_GAN_CFD/model_ckpt/model_D.keras')
	g_model = load_model('./training_GAN_CFD/model_ckpt/model_G.keras')
	gan_model = load_model('./training_GAN_CFD/model_ckpt/model_GAN.keras')
else:
	# define the models
	d_model = define_discriminator(image_shape) #(None, 4, 32, 1)
	g_model = define_generator(image_shape)	#(None, 64, 512, 1)
	gan_model = define_gan(g_model, d_model, image_shape)	#[(None, 4, 32, 1), (None, 64, 512, 1)]

# train model
# print(d_model.summary())
# print(g_model.summary())
# print(gan_model.summary())
train(d_model, g_model, gan_model)


