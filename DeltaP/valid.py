from numpy import *
from pandas import *
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import StrMethodFormatter
import skfmm
import cv2
import os
import csv
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_data


dir_name = './DeltaP/'
model_name = dir_name + '/training_DeltaP/model_ckpt/'
eval_name = dir_name + '/evaluation_valid/'

def plot_image(savepath, var, pretext, fieldname, flag, vmin=None, vmax=None, delta_p=0):
	"""
	Generates and saves images for the given data array, highlighting the maximum value if applicable.
	"""
	if flag == 1:
		labeltxt = 'SDF Boundary'
	elif flag == 2:
		labeltxt = 'Pressure (Pa)'
	elif flag == 3:
		labeltxt = 'U (m/s)'
	elif flag == 4:
		labeltxt = '% Error'

	var[np.isinf(var)] = np.nan
	var = np.clip(var, vmin, vmax)

	Z, Y = np.meshgrid(np.linspace(0, 50, var.shape[1]), np.linspace(0, 4, var.shape[0]))

	fig, ax = plt.subplots()
	ax.set_aspect('equal', adjustable='box')
	contour = ax.contourf(Z, Y, var, 100, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)

	fig.colorbar(contour, ax=ax, label=labeltxt,orientation='horizontal')

	if "delta_pressure" in pretext.lower():
		ax.text(
			-1, 10, f'Abs Error: {delta_p:.3f}', 
			color='white', fontsize=11, fontweight=550,
			ha='left', va='bottom', 
			bbox=dict(facecolor='black', edgecolor='none', alpha=0.8)  # Set transparency with alpha
		)
	# Save the plot
	plt.savefig(savepath + pretext + fieldname + '.png')
	plt.close(fig)

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, X_realA, patch_shape):
	# generate fake instance
	X = []
	for i in range(len(X_realA)):
		x = g_model.predict(X_realA[[i]])
		x = np.asarray(x[0])
		X.append(x)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	X = np.asarray(X).astype('float32')
	return X, y


# load model using tf.keras
model = load_model('./'+model_name+'model_G.keras', compile=False)


processed_folder = "./130_case/processed/"
bnd_array, sflow_P_array, sflow_U_array, DeltaP_array = load_data(processed_folder)

train_idx = np.loadtxt(dir_name + '/training_DeltaP/train_indices.txt', dtype=int)
valid_idx = np.loadtxt(dir_name + '/training_DeltaP/valid_indices.txt', dtype=int)


x_train_1 = []
y_train_1 = []
x_test_1 = []
y_test_1 = []
global p_valid_1 
p_valid_1 = []

for i in train_idx:
  x_train_1.append(bnd_array[i])
  y_train_1.append(DeltaP_array[i])

for i in valid_idx:
  x_test_1.append(bnd_array[i])
  y_test_1.append(DeltaP_array[i])
  p_valid_1.append(sflow_P_array[i])

x_train_1 = np.array(x_train_1)
y_train_1 = np.array(y_train_1)
x_test_1 = np.array(x_test_1)
y_test_1 = np.array(y_test_1)

train_size = len(x_train_1)
test_size = len(x_test_1)

x_train_1 = np.asarray(x_train_1).astype('float32')
y_train_1 = np.asarray(y_train_1).astype('float32')

x_train_1 = np.expand_dims(x_train_1,axis=3)
y_train_1 = np.expand_dims(y_train_1,axis=-1)


x_test_1 = np.asarray(x_test_1).astype('float32')
y_test_1 = np.asarray(y_test_1).astype('float32')

x_test_1 = np.expand_dims(x_test_1,axis=3)
y_test_1 = np.expand_dims(y_test_1,axis=-1)


global P_max
P_max = y_train_1.max()
y_train_1 = y_train_1 / P_max
y_test_1 = y_test_1 / P_max

data_size = len(x_test_1)

#Y_max = 50.9  # Temp 20-40
#Y_min = 4.75
#y_test_1 = (y_test_1 - T_min) / (T_max - T_min)
# Y_max = 25.8  # UX 10
# Y_min = -10.9
# model_prediction(model, x_test_1, y_test_2, T_max, T_min, i, 'UX')

y_error = []
Error_list = []
Error_mean = np.ones(64)
field = 'UX'
flag = 3
sum_error_avg = 0

file_path = eval_name + 'error_test_UX_ibm_cylinder.csv'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(eval_name + 'error_test_UX_ibm_cylinder.csv', "w") as csv_file:
  writer = csv.writer(csv_file)
  writer.writerow(['Test ID', 'Abs Error (Pa)', 'Relative (%)'])
  for ix in range(data_size):
    X_fakeB, _ = generate_fake_samples(model, x_test_1[[ix]], 1)
	
    X = x_test_1[ix]
    X = np.squeeze(X, axis=2)

    P = p_valid_1[ix]

    y = X_fakeB[0] * P_max

    Y = y_test_1[ix] * P_max

    y_error = abs(Y - y)
    print(f'Ground truth: {Y[0]}')
    print(f'Predicted: {y[0]}')
    print(f'Abs Error: {y_error[0]}')
    vmin = min(y_error.min(), Y.min(), y.min())
    vmax = max(y_error.max(), Y.max(), y.max())

    plot_image(eval_name, P, str(ix) + '_Delta_Pressure_', field, 2, vmin=P.min(), vmax=P.max(), delta_p=y_error[0])

    Abs_error = y_error[0]
    Relative_error = y_error[0]/Y[0]*100

    writer.writerow([str(ix),str(Abs_error),str(Relative_error)])
    

