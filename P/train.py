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
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models import *
from utils import generate_fake_samples, generate_real_samples, load_data, plot_image, summarize_performance, str2bool
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Input option for training GAN CFD model')
parser.add_argument("-c", "--checkpoint", default=False, help="Whether load checkpoint for training", type=str2bool)
parser.add_argument("-e", "--epocheval", default=1, help="Number of epoch for each evaluation", type=int)
parser.add_argument("-n", "--numepoch", default=10000, help="Number of epoch for training iteration", type=int)
parser.add_argument("-t", "--trainpath", default='30_case_duyanh/processed/', help="training directory input", type=str)
# parser.add_argument("-t", "--trainpath", default='130_case_test/', help="training directory input", type=str)
parser.add_argument("-d", "--directory", default='P/training_P/', help="training directory output", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
# train pix2pix models
def train(g_model, n_epochs=args.numepoch, batch_size=1):
	n_patch = d_model.output_shape[1]
	flag = 2

	# determine the output square shape of the discriminator
	bnd_array, sflow_P_array, sflow_U_array, DeltaP_array = load_data(args.trainpath)
	

	train_idx, valid_idx = train_test_split(range(len(bnd_array)), test_size=0.1, random_state=42)
	x_train_1 = [bnd_array[i] for i in train_idx]
	x_valid_1 = [bnd_array[i] for i in valid_idx]
	y_train_1 = [sflow_P_array[i] for i in train_idx]
	y_valid_1 = [sflow_P_array[i] for i in valid_idx]	

	if not os.path.exists(dir_name):
		os.makedirs(dir_name)

	np.savetxt(dir_name +'train_indices.txt', train_idx, fmt='%d')
	np.savetxt(dir_name + 'valid_indices.txt', valid_idx, fmt='%d')
	# train_idx = np.loadtxt('train_indices.txt', dtype=int)
	# valid_idx = np.loadtxt('valid_indices.txt', dtype=int)
	train_size = len(x_train_1)
	valid_size = len(x_valid_1)

	x_train_1 = np.asarray(x_train_1).astype('float32')
	y_train_1 = np.asarray(y_train_1).astype('float32')

	x_train_1 = np.expand_dims(x_train_1,axis=3)
	y_train_1 = np.expand_dims(y_train_1,axis=3)


	x_valid_1 = np.asarray(x_valid_1).astype('float32')
	y_valid_1 = np.asarray(y_valid_1).astype('float32')

	x_valid_1 = np.expand_dims(x_valid_1,axis=3)
	y_valid_1 = np.expand_dims(y_valid_1,axis=3)


	# normalize
	global P_max
	P_max = y_train_1.max()
	y_train_1 = y_train_1 / P_max
	y_valid_1 = y_valid_1 / P_max

	print('x1 train shape',x_train_1.shape)
	print(f'y1 shape: {y_train_1.shape}, P min = {y_train_1.min()}, P max = {y_train_1.max()}')

	print('x1 valid shape',x_valid_1.shape)
	print(f'y1 shape: {y_valid_1.shape}, P min = {y_valid_1.min()}, P max = {y_valid_1.max()}')

	
 
	# calculate the number of batches per training epoch
	bat_per_epo = int(train_size / batch_size)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	idx_choice = [i for i in range(train_size)]
	d_1 = d_2 = g_ = 0
	count_epoch = 0
	total_d1 = []
	total_d2 = []
	total_g  = []
	total_val = []
	min_eval = -1

	if chkpt == True:
		try:
			with open(dir_name + 'steps.txt', 'r') as f:
				start_step = int(f.read().strip())
			start_step = (start_step // train_size) * train_size
		except FileNotFoundError:
			start_step = 0
	else:
		start_step = 0

	init_t = time.time()
	# manually enumerate epochs
	for i in range(start_step, n_steps): # batch count
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
		[X_realA, X_realB], y_real = generate_real_samples(x_train_1, y_train_1, idx, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples 
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated sample
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, g_loss1, g_loss2 = gan_model.train_on_batch([X_realA], [y_real, X_realB]) 
		 
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		d_1 += d_loss1
		d_2 += d_loss2
		g_  += g_loss

		#print_memory_usage()
		
		with open(dir_name + 'steps.txt', 'w') as f:
				f.write(str(i + 1))

		# summarize model performance
		if (ix+1) % (bat_per_epo) == 0:
			print('epoch time: {}'.format(time.time()-init_t))
			idx_choice = [i for i in range(train_size)]
			val_loss = 0
			for k in range(valid_size):
				X_validA = x_valid_1[[k]]
				X_validB = y_valid_1[[k]]
				#tensorflow==2.18.0
				#val_,_ , _, _ = gan_model.test_on_batch([X_validA], [y_real[[0]], X_validB])
				#tensorflow==2.17.0 or 2.16.1
				val_, _, _ = gan_model.test_on_batch([X_validA], [y_real[[0]], X_validB])
				val_loss += val_
			val_loss = val_loss / valid_size
			
			if min_eval == -1 or val_loss < min_eval:
				# save the generator model
				min_eval = val_loss
				# filename2 = dir_name + "model_eval.keras"
				# save_model(g_model, filename2, overwrite=True, include_optimizer=True) 
				# print('---->Saved model: %s' % (filename2))
			#args.epocheval
			if (i+1) % (bat_per_epo * args.epocheval) == 0:
				summarize_performance(args.directory, i, flag, g_model, x_valid_1, y_valid_1, valid_size, 'P', scale=P_max)
			count_epoch += 1
			total_d1.append(d_1 / train_size)
			total_d2.append(d_2 / train_size)
			total_g.append(g_ / train_size)
			total_val.append(val_loss)
			d_1 = d_2 = g_ = 0

			fig1 = plt.figure()
			fig1 = plt.subplots()
			plt.plot(range(count_epoch), total_d1, "-r", label='D1 loss')
			plt.plot(range(count_epoch), total_d2, "-g", label='D2 loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Training loss")
			plt.savefig(dir_name + 'training_D_loss.png')

			fig2 = plt.figure()
			fig2 = plt.subplots()
			plt.plot(range(count_epoch), total_g, "-b", label='G loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Training loss")
			plt.savefig(dir_name + 'training_G_loss.png')

			fig3 = plt.figure()
			fig3 = plt.subplots()
			plt.plot(range(count_epoch), total_val, "-y", label='Val loss')
			plt.legend(loc="upper center")
			plt.xlabel("Epoch number")
			plt.ylabel("Validation loss")
			plt.savefig(dir_name + 'validation_loss.png')
			plt.close('all')

			if (i+1) % (bat_per_epo) == 0:
				dir_model_name = dir_name + 'model_ckpt/'
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
dir_name = './P/training_P/'

if chkpt == True:
	d_model = load_model(dir_name + 'model_ckpt/model_D.keras')
	g_model = load_model(dir_name + 'model_ckpt/model_G.keras')
	gan_model = load_model(dir_name + 'model_ckpt/model_GAN.keras')
else:
	d_model = define_discriminator(image_shape)
	g_model = define_generator(image_shape)
	gan_model = define_gan(g_model, d_model, image_shape)

# train model
# print(g_model.summary())
train(g_model)