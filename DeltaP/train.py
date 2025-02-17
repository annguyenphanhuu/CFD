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
from utils import generate_fake_samples, generate_real_samples
import argparse
from sklearn.model_selection import train_test_split
from utils import generate_fake_samples, generate_real_samples, load_data, plot_image, summarize_performance, str2bool

parser = argparse.ArgumentParser(description='Input option for training GAN CFD model')
parser.add_argument("-c", "--checkpoint", default=False, help="Whether load checkpoint for training", type=str2bool)
parser.add_argument("-e", "--epocheval", default=1, help="Number of epoch for each evaluation", type=int)
parser.add_argument("-n", "--numepoch", default=10000, help="Number of epoch for training iteration", type=int)
parser.add_argument("-t", "--trainpath", default='30_case_duyanh/processed/', help="training directory input", type=str)
parser.add_argument("-d", "--directory", default='DeltaP/training_DeltaP/', help="training directory output", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
# train pix2pix models
def train(g_model, n_epochs=args.numepoch, batch_size=1):
	flag = 5

	bnd_array, sflow_P_array, sflow_U_array, DeltaP_array = load_data(args.trainpath)

	bnd_array = np.array(bnd_array)
	sflow_P_array = np.array(sflow_P_array)
	DeltaP_array = np.array(DeltaP_array)

	train_idx, valid_idx = train_test_split(range(len(bnd_array)), test_size=0.1, random_state=42)
	x_train_1 = [bnd_array[i] for i in train_idx]
	x_valid_1 = [bnd_array[i] for i in valid_idx]
	y_train_1 = [DeltaP_array[i] for i in train_idx]
	y_valid_1 = [DeltaP_array[i] for i in valid_idx]

	p_valid = [sflow_P_array[i] for i in valid_idx]

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
	p_valid = np.asarray(p_valid).astype('float32')

	x_train_1 = np.expand_dims(x_train_1,axis=3)
	y_train_1 = np.expand_dims(y_train_1,axis=-1)


	x_valid_1 = np.asarray(x_valid_1).astype('float32')
	y_valid_1 = np.asarray(y_valid_1).astype('float32')

	x_valid_1 = np.expand_dims(x_valid_1,axis=3)
	y_valid_1 = np.expand_dims(y_valid_1,axis=-1)

	# normalize data to range [0,1]
	global P_max
	P_max = y_train_1.max()
	y_train_1 = y_train_1 / P_max
	y_valid_1 = y_valid_1 / P_max

	print('x1 train shape',x_train_1.shape)
	print(f'y1 shape: {y_train_1.shape}, P min = {y_train_1.min()}, P max = {y_train_1.max()}')
	# print(f'y2 shape: {y_train_2.shape}, U min = {y_train_2.min()}, U max = {y_train_2.max()}')
	print('x1 valid shape',x_valid_1.shape)
	print(f'y1 shape: {y_valid_1.shape}, P min = {y_valid_1.min()}, P max = {y_valid_1.max()}')
	# print(f'y2 shape: {y_valid_2.shape}, U min = {y_valid_2.min()}, U max = {y_valid_2.max()}')
	
 
	# calculate the number of batches per training epoch
	bat_per_epo = int(train_size / batch_size)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	idx_choice = [i for i in range(train_size)]
	g_ = 0
	count_epoch = 0
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
		[X_realA, X_realB], y_real = generate_real_samples(x_train_1, y_train_1, idx, 1)
		#print('debug X_labelA: '+str(X_labelA.shape))
		g_loss = g_model.train_on_batch([X_realA], [X_realB])
		print('>%d, g[%.3f]' % (i+1, g_loss))
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
				val_ = g_model.test_on_batch([X_validA], [X_validB])
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
				summarize_performance(args.directory, i, flag, g_model, x_valid_1, y_valid_1, valid_size, 'P', P_max, p=p_valid)
			count_epoch += 1
			total_g.append(g_ / train_size)
			total_val.append(val_loss)
			g_ = 0


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
				print('---->Saved checkpoint: %s' % (filename1))

			init_t = time.time()
            
# LOAD TRAINING CHECKPOINTS
image_shape = (64,512,1)
chkpt = args.checkpoint
dir_name = './DeltaP/training_DeltaP/'

if chkpt == True:
	g_model = load_model(dir_name + 'model_ckpt/model_G.keras')
else:
	g_model = define_cnn_model(image_shape)

# train model
# print(g_model.summary())
train(g_model)


