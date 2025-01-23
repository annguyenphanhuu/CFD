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
from utils import plot_image

dir_name = './P/'
model_name = dir_name + '/training_P/model_ckpt/'
eval_name = dir_name + '/evaluation_test/'

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
bnd_array = []
sflow_P_array = []

processed_folder = "./100_case_si/processed/"
for file_path in os.listdir(processed_folder):
  with open(processed_folder + file_path, 'r') as filename:
    file = csv.DictReader(filename)
    P_array = []
    U_array = []
    pts = np.ones((64,512))
    p = np.zeros((64,512))
    u = np.zeros((64,512))

    # Extract data from CSV columns
    for col in file:
      P_array.append(float(col['p_convert']))
      U_array.append(float(col['Umean_center']))
    # Map data into the domain (4x50 mm)
    for i in range(64):
      for j in range(512):
        p[i, j] = P_array[i * 512 + j]
        u[i, j] = U_array[i * 512 + j]
        if P_array[i * 512 + j] <= 0:
          pts[i, j] = 1
        else:
          pts[i, j] = -1
    sflow_P_array.append(p)
    # Compute SDF (Signed Distance Function)

    phi = pts
    d_x = 4 / 64
    d = skfmm.distance(phi, d_x)

    bnd = np.array(d)
    bnd_array.append(bnd)

train_idx = np.loadtxt(dir_name + '/training_P/train_indices.txt', dtype=int)
valid_idx = np.loadtxt(dir_name + '/training_P/valid_indices.txt', dtype=int)

y_train_1 = []
for i in train_idx:
  y_train_1.append(sflow_P_array[i])
y_train_1 = np.array(y_train_1)

P_max = y_train_1.max()
y_train_1 = y_train_1 / (2* P_max)

bnd_array = []
sflow_P_array = []

processed_folder = "./30_case_duyanh/processed/"
for file_path in os.listdir(processed_folder):
  with open(processed_folder + file_path, 'r') as filename:
    file = csv.DictReader(filename)
    P_array = []
    U_array = []
    pts = np.ones((64,512))
    p = np.zeros((64,512))
    u = np.zeros((64,512))

    # Extract data from CSV columns
    for col in file:
      P_array.append(float(col['p_convert']))
      U_array.append(float(col['Umean_center']))
    # Map data into the domain (4x50 mm)
    for i in range(64):
      for j in range(512):
        p[i, j] = P_array[i * 512 + j]
        u[i, j] = U_array[i * 512 + j]
        if P_array[i * 512 + j] <= 0:
          pts[i, j] = 1
        else:
          pts[i, j] = -1
    sflow_P_array.append(p)
    # Compute SDF (Signed Distance Function)

    phi = pts
    d_x = 4 / 64
    d = skfmm.distance(phi, d_x)

    bnd = np.array(d)
    bnd_array.append(bnd)

test_size = len(bnd_array)
x_test_1 = bnd_array[-test_size:]
y_test_1 = sflow_P_array[-test_size:]


# x_train_1 = np.asarray(x_train_1).astype('float32')
# y_train_1 = np.asarray(y_train_1).astype('float32')
# y_train_2 = np.asarray(y_train_2).astype('float32')
# x_train_1 = np.expand_dims(x_train_1,axis=3)
# y_train_1 = np.expand_dims(y_train_1,axis=3)
# y_train_2 = np.expand_dims(y_train_2,axis=3)

x_test_1 = np.asarray(x_test_1).astype('float32')
y_test_1 = np.asarray(y_test_1).astype('float32')

x_test_1 = np.expand_dims(x_test_1,axis=3)
y_test_1 = np.expand_dims(y_test_1,axis=3)

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
  writer.writerow(['Test ID', 'Error max (m/s)', 'Error max (%)', 'Error mean (m/s)', 'Cosine Similarity'])
  for ix in range(data_size):
    X_fakeB, _ = generate_fake_samples(model, x_test_1[[ix]], 1)
	
    X = x_test_1[ix]
    X = np.squeeze(X, axis=2)

    y = X_fakeB[0] * (2 * P_max)
    y = np.squeeze(y, axis=2)

    Y = y_test_1[ix]
    Y = np.squeeze(Y, axis=2)

    y_error = abs(Y-y)  

    vmin = min(y_error.min(), Y.min(), y.min())
    vmax = max(y_error.max(), Y.max(), y.max())
    #plot_image(eval_name, X,str(ix)+'_Boundary_',field,1)
    plot_image(eval_name, Y,str(ix)+'_CFD_',field,flag, vmin=vmin, vmax=vmax)
    plot_image(eval_name, y,str(ix)+'_Predict_',field,flag, vmin=vmin, vmax=vmax)
    plot_image(eval_name, y_error,str(ix)+'_error_abs_', field, 3, vmin=vmin, vmax=vmax)
    plot_image(eval_name, y_error/Y*100,str(ix)+'_error_%_', field, 4, vmin=0, vmax=100)
    
    sum_err = np.sum(y_error)
    err_avg = sum_err/(64*512)
    sum_error_avg += err_avg

    Error_mean = np.asarray(y_error.max(axis=0))
    Error_max = Error_mean.max()

    max_index = np.argmax(y_error)
    max_index = np.unravel_index(max_index, y_error.shape)
    Error_percentage_max = (y_error[max_index] / Y[max_index]) * 100

    # cos_sim = np.mean([
    #     cosine_similarity([y[i]], [Y[i]])[0][0] 
    #     for i in range(len(y)) 
    #     if not all(val == 0 for val in Y[i])
    # ])
    cosine_arr = []
    for i in range(len(y)):
      if not np.all(Y[i] == 0):
         cosine_arr.append(cosine_similarity([y[i]], [Y[i]])[0][0])
    cos_sim = np.mean(cosine_arr)

    print('Error average: ',err_avg)
    print('Error max:',Error_max)
    print('Error percentage max:',Error_percentage_max)
    print('Cosine Similarity: ',cos_sim)
    writer.writerow([str(ix),str(Error_max),str(Error_percentage_max),str(err_avg),str(cos_sim)])
    
    Error_list.append(Error_mean)

errors_avg = sum_error_avg/data_size
Error_list = np.asarray(Error_list).astype('float32')
test_list = [i for i in range(data_size)]
Z = np.arange(0, 0.05, 0.05/512)
Y = np.asarray(test_list).astype('float32')
Z, Y = np.meshgrid(Z, Y)

# Plot the surface.
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(Z, Y, Error_list, cmap=plt.cm.rainbow, linewidth=0, antialiased=False)

# Customize the z axis
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
fmt = StrMethodFormatter('{x:.0f}')
ax.zaxis.set_major_formatter(fmt)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.4, aspect=10, label='Test cGAN UX Cylinder - Rel.Err (%)')
ax.set_xlabel(" X (mm)")
ax.set_ylabel("Test ID")
plt.savefig(eval_name + 'Validation_UX_ibm_cylinder_'+str(round(errors_avg,2))+'.png')
plt.show()
print('Total average relative error: ',errors_avg)