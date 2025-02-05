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
from utils import plot_image, load_data

dir_name = './P/'
model_name = dir_name + '/training_P/model_ckpt/'
eval_name = dir_name + '/evaluation_valid/'

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, cond, X_realA, patch_shape):
  # generate fake instance
  X = []
  for i in range(len(X_realA)):
    cond = np.array(cond, dtype=np.float32).reshape(1, -1)
    x = g_model.predict([X_realA[[i]], cond])
    x = np.asarray(x[0])
    X.append(x)
  # create 'fake' class labels (0)
  y = np.zeros((len(X), patch_shape, patch_shape * 8, 1))
  X = np.asarray(X).astype('float32')
  return X, y


# load model using tf.keras
model = load_model('./'+model_name+'model_G.keras', compile=False)

processed_folder = "./100_case_si_test/processed/"
bnd_array, sflow_P_array, sflow_U_array, DeltaP_array, cond_array = load_data(processed_folder)

train_idx = np.loadtxt(dir_name + '/training_P/train_indices.txt', dtype=int)
valid_idx = np.loadtxt(dir_name + '/training_P/valid_indices.txt', dtype=int)


x_train_1 = []
y_train_1 = []
x_test_1 = []
y_test_1 = []
for i in train_idx:
  x_train_1.append(bnd_array[i])
  y_train_1.append(sflow_P_array[i])
x_test_1 = []
for i in valid_idx:
  x_test_1.append(bnd_array[i])
  y_test_1.append(sflow_P_array[i])

x_train_1 = np.array(x_train_1)
y_train_1 = np.array(y_train_1)
x_test_1 = np.array(x_test_1)
y_test_1 = np.array(y_test_1)

P_max = y_train_1.max()
y_train_1 = y_train_1 / P_max

x_test_1 = np.asarray(x_test_1).astype('float32')
y_test_1 = np.asarray(y_test_1).astype('float32')

x_test_1 = np.expand_dims(x_test_1,axis=3)
y_test_1 = np.expand_dims(y_test_1,axis=3)

data_size = len(x_test_1)

y_error = []
Error_list = []
Error_mean = np.ones(64)
field = 'UX'
flag = 2
sum_error_avg = 0

file_path = eval_name + 'error_test_UX_ibm_cylinder.csv'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open(eval_name + 'error_test_UX_ibm_cylinder.csv', "w") as csv_file:
  writer = csv.writer(csv_file)
  writer.writerow(['Test ID', 'Error max (Pa)', 'Error max (%)', 'Error mean (Pa)', 'Cosine Similarity'])
  for ix in range(data_size):
    X_fakeB, _ = generate_fake_samples(model, cond_array[ix], x_test_1[[ix]], 1)
	
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
    plot_image(eval_name, y_error,str(ix)+'_error_abs_', field, flag, vmin=vmin, vmax=vmax)
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