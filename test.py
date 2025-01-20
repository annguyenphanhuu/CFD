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

#model_name = 'model_ckpt_T_ibm_256/'

model_name = './training_GAN_CFD/model_ckpt/'

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
    elif flag == 4:
        labeltxt = '% Error'

    var[np.isinf(var)] = np.nan
    var = np.clip(var, vmin, vmax)

    Z, Y = np.meshgrid(np.linspace(0, 50, 512), np.linspace(0, 4, 64))
    fig, ax = plt.subplots()
    # im = ax.imshow(var, vmin=vmin, vmax=vmax, origin='lower',
    #                extent=[Z.min(), Z.max(), Y.min(), Y.max()])
    ax.set_aspect('equal', adjustable='box')
    contour = ax.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, label=labeltxt)

    if "error_abs" in pretext.lower():
        total_value = np.sum(var)
        ax.text(0.05, 5, f'Total: {total_value:.2f}', transform=ax.transAxes,
                fontsize=12, color='white', backgroundcolor='black',
                verticalalignment='top')

    # Save the plot
    plt.savefig('./model_evaluation/' + pretext + fieldname + '.png')
    plt.close(fig)
    
# load model using tf.keras
model = load_model('./'+model_name+'model_G.keras', compile=False)
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
    #pts = cv2.resize(pts, (512, 64), interpolation=cv2.INTER_NEAREST)
    phi = pts
    d_x = 4 / 64
    d = skfmm.distance(phi, d_x)

    bnd = np.array(d)
    bnd_array.append(bnd)

train_size = int(0.85 * len(bnd_array))
test_size = len(bnd_array) - train_size

x_train_1 = bnd_array[:train_size]
x_test_1 = bnd_array[-test_size:]
y_train_1 = sflow_P_array[:train_size]
y_test_1 = sflow_P_array[-test_size:]
y_train_2 = sflow_U_array[:train_size]
y_test_2 = sflow_U_array[-test_size:]

x_train_1 = np.asarray(x_train_1).astype('float32')
y_train_1 = np.asarray(y_train_1).astype('float32')
y_train_2 = np.asarray(y_train_2).astype('float32')
x_train_1 = np.expand_dims(x_train_1,axis=3)
y_train_1 = np.expand_dims(y_train_1,axis=3)
y_train_2 = np.expand_dims(y_train_2,axis=3)

x_test_1 = np.asarray(x_test_1).astype('float32')
y_test_1 = np.asarray(y_test_1).astype('float32')
y_test_2 = np.asarray(y_test_2).astype('float32')
x_test_1 = np.expand_dims(x_test_1,axis=3)
y_test_1 = np.expand_dims(y_test_1,axis=3)
y_test_2 = np.expand_dims(y_test_2,axis=3)
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
sum_error = 0

file_path = './model_evaluation/error_test_UX_ibm_cylinder.csv'
os.makedirs(os.path.dirname(file_path), exist_ok=True)
with open('./model_evaluation/error_test_UX_ibm_cylinder.csv', "w") as csv_file:
  writer = csv.writer(csv_file)
  writer.writerow(['Test ID', 'Error max (%)', 'Error mean'])
  for ix in range(data_size):
    X_fakeB, _ = generate_fake_samples(model, x_test_1[[ix]], 1)
	
    X = x_test_1[ix]
    X = np.squeeze(X, axis=2)
    y = X_fakeB[0]
    y = np.squeeze(y, axis=2)
    #y = y * (Y_max - Y_min) + Y_min
    Y = y_test_2[ix]
    Y = np.squeeze(Y, axis=2)
    y_error = abs(Y-y)  
    vmin = min(y_error.min(), Y.min(), y.min())
    vmax = max(y_error.max(), Y.max(), y.max())
    #plot_image(X,str(ix)+'_Boundary_',field,1)
    plot_image(Y,str(ix)+'_CFD_',field,flag, vmin=vmin, vmax=vmax)
    plot_image(y,str(ix)+'_Predict_',field,flag, vmin=vmin, vmax=vmax)
    plot_image(y_error,str(ix)+'_error_abs_', field, 3, vmin=vmin, vmax=vmax)
    plot_image(y_error/Y*100,str(ix)+'_error_%_', field, 4, vmin=0, vmax=100)
    
    ymax = Y.max()
    ymean = np.mean(y_test_1[ix])
    print('ymax', ymax, ' at test ID: ',str(ix))

    sum_err = np.sum(y_error)
    err_avg = sum_err/(64*512)
    print('Error average: ',err_avg)
    sum_error += err_avg

    # y_error = y_error*100/ymax
    Error_mean = np.asarray(y_error.max(axis=0))
    Error_max = Error_mean.max()

    max_index = np.argmax(y_error)
    max_index = np.unravel_index(max_index, y_error.shape)
    Error_percentage_max = (y_error[max_index] / Y[max_index]) * 100

    Error_avg = np.sum(y_error)/(64*512)

    print('Error max:',Error_max)
    print('Error percentage max:',Error_percentage_max)
    print('Error average:',Error_avg)
    writer.writerow([str(ix),str(Error_percentage_max),str(Error_avg)])
    
    Error_list.append(Error_mean)

errors_avg = sum_error/data_size
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
ax.set_xlabel(" X (m)")
ax.set_ylabel("Test ID")
plt.savefig('./model_evaluation/Validation_UX_ibm_cylinder_'+str(round(errors_avg,2))+'.png')
plt.show()
print('Total average relative error: ',errors_avg)