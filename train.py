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
from tensorflow.keras.activations import swish
import argparse

parser = argparse.ArgumentParser(description='Input option for training GAN CFD model')
parser.add_argument("-c", "--checkpoint", default=False, help="Whether load checkpoint for training", type=Boolean)
parser.add_argument("-e", "--epocheval", default=5, help="Number of epoch for each evaluation", type=int)
parser.add_argument("-n", "--numepoch", default=3000, help="Number of epoch for training iteration", type=int)
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
    out_image = Activation('sigmoid')(g)
    
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
        # Read data
        data = pd.read_csv(data_file)
        
        # Check exist column
        required_columns = ['y_center', 'z_center', 'p_center', 'Umean_center']
        if not all(col in data.columns for col in required_columns):
            print("Missing required columns in CSV file!")
            return []
        
        # Get column's value
        xlist = data['y_center'].values
        ylist = data['z_center'].values
        temp = data['p_center'].values
        vx = data['Umean_center'].values
        
        # Process
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
                # Interpolate
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

def plot_image(var, pretext, fieldname, flag, vmin=None, vmax=None):
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
    fig, ax = plt.subplots()
    im = ax.imshow(var, vmin=vmin, vmax=vmax, origin='lower',
                   extent=[Z.min(), Z.max(), Y.min(), Y.max()])
    ax.set_aspect('equal', adjustable='box')
    contour = ax.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, label=labeltxt)

    if "error" in pretext.lower():
        total_value = np.sum(var)
        ax.text(0.05, 5, f'Total: {total_value:.2f}', transform=ax.transAxes,
                fontsize=12, color='white', backgroundcolor='black',
                verticalalignment='top')

    # Save the plot
    plt.savefig(args.directory + pretext + fieldname + '.png')
    plt.close(fig)
    
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, x_test_1, y_test_2, test_size, field):
    # Generate a batch of fake samples
    ix = random.randint(0, test_size - 1)
    X_fakeB, _ = generate_fake_samples(g_model, x_test_1[[ix]], 1)

    X = x_test_1[ix]
    X = np.squeeze(X, axis=2)
    y = X_fakeB[0]
    y = np.squeeze(y, axis=2)
    Y = y_test_2[ix]
    Y = np.squeeze(Y, axis=2)

    y_error = abs(Y - y)

    print('X shape', X.shape)
    print('Y shape', Y.shape)
    print('y shape', y.shape)

    # Plot each variable with its own min and max
    plot_image(X, str(step + 1) + '_Boundary_', field, 1, vmin=X.min(), vmax=X.max())
    plot_image(Y, str(step + 1) + '_CFD_', field, 3, vmin=Y.min(), vmax=Y.max())
    plot_image(y, str(step + 1) + '_Predict_', field, 3, vmin=y.min(), vmax=y.max())

    # Calculate and plot the error
    plot_image(y_error, str(step + 1) + '_error_abs_', field, 3, vmin=y_error.min(), vmax=y_error.max())
	

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
 
	# calculate the number of batches per training epoch
	bat_per_epo = int(data_size / batch_size)
	# calculate the number of training iterations
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
	# manually enumerate epochs
	for i in range(n_steps): # batch count
		gc.collect()
		# get dataset index for each epoch
		ix = i % bat_per_epo
		idx = []
		# random index for dataset training
		for k in range(batch_size):
			index = choice(idx_choice)
			idx.append(index)
			idx_choice.remove(index)
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(x_train_1, y_train_2, idx, n_patch)
		#print('debug X_labelA: '+str(X_labelA.shape))
		# generate a batch of fake samples
		#tf.data.Dataset
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
  		# update discriminator for real samples 
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated sample
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		#tensorflow==2.18.0
		#g_loss, _, g_loss1, g_loss2 = gan_model.train_on_batch([X_realA], [y_real, X_realB])
		#tensorflow==2.17.0	 or 2.16.1
		g_loss, g_loss1, g_loss2 = gan_model.train_on_batch([X_realA], [y_real, X_realB]) 
		 
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		d_1 += d_loss1
		d_2 += d_loss2
		g_  += g_loss

		#print_memory_usage()


		# summarize model performance
		if (ix+1) % (bat_per_epo) == 0:
			print('epoch time: {}'.format(time.time()-init_t))
			idx_choice = [i for i in range(data_size)]
			val_loss = 0
			for k in range(test_size):
				X_testA = x_test_1[[k]]
				X_testB = y_test_2[[k]]
				#tensorflow==2.18.0
				#val_,_ , _, _ = gan_model.test_on_batch([X_testA], [y_real[[0]], X_testB])
				#tensorflow==2.17.0 or 2.16.1
				val_, _, _ = gan_model.test_on_batch([X_testA], [y_real[[0]], X_testB])
				val_loss += val_
			val_loss = val_loss / test_size
			
			if min_eval == -1 or val_loss < min_eval:
				#Create training folder
				dir_name = './training_GAN_CFD/'
				if not os.path.exists(dir_name):
					os.makedirs(dir_name)
				# save the generator model
				min_eval = val_loss
				filename2 = dir_name + "model_eval.keras"
				save_model(g_model, filename2, overwrite=True, include_optimizer=True) 
				print('---->Saved model: %s' % (filename2))
			#args.epocheval
			if (i+1) % (bat_per_epo * args.epocheval) == 0:
				summarize_performance(i, g_model, x_test_1, y_test_2, test_size, 'UX')
			count_epoch += 1
			total_d1.append(d_1 / data_size)
			total_d2.append(d_2 / data_size)
			total_g.append(g_ / data_size)
			total_val.append(val_loss)
			d_1 = d_2 = g_ = 0

			fig1 = plt.figure()
			fig1 = plt.subplots()
			plt.plot(range(count_epoch), total_d1, "-r", label='D1 loss')
			plt.plot(range(count_epoch), total_d2, "-g", label='D2 loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Training loss")
			plt.savefig('./training_GAN_CFD/training_D_loss.png')

			fig2 = plt.figure()
			fig2 = plt.subplots()
			plt.plot(range(count_epoch), total_g, "-b", label='G loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Training loss")
			plt.savefig('./training_GAN_CFD/training_G_loss.png')

			fig3 = plt.figure()
			fig3 = plt.subplots()
			plt.plot(range(count_epoch), total_val, "-y", label='Val loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Validation loss")
			plt.savefig('./training_GAN_CFD/validation_loss.png')
			plt.close('all')

			if (i+1) % (bat_per_epo * 10) == 0:
				dir_model_name = './training_GAN_CFD/model_ckpt/'
				if not os.path.exists(dir_model_name):
					os.makedirs(dir_model_name)
				filename1 = dir_model_name+ "model_G.keras"
				save_model(g_model, filename1, overwrite=True, include_optimizer=True)
				filename2 = dir_model_name+ "model_D.keras"
				save_model(d_model, filename2, overwrite=True, include_optimizer=True)
				filename3 = dir_model_name+ "model_GAN.keras"
				save_model(gan_model, filename3, overwrite=True, include_optimizer=True)
				print('---->Saved checkpoint: %s' % (filename1))
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


