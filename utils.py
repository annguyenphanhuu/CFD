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

def plot_image(savepath, var, pretext, fieldname, flag, vmin=None, vmax=None):
    """
    Generates and saves images for the given data array, highlighting the maximum value if applicable.
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

    Z, Y = np.meshgrid(np.linspace(0, 50, var.shape[1]), np.linspace(0, 4, var.shape[0]))
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    contour = ax.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)
    fig.colorbar(contour, ax=ax, label=labeltxt)

    if "error_abs" in pretext.lower():
        total_value = np.nansum(var)
        ax.text(0.05, 4.05, f'Total: {total_value:.2f}', transform=ax.transAxes,
                fontsize=12, color='white', backgroundcolor='black',
                verticalalignment='top')

        # Find the maximum value and its index
        max_index = np.nanargmax(var)
        max_coords = np.unravel_index(max_index, var.shape)
        max_z, max_y = Z[max_coords[0], max_coords[1]], Y[max_coords[0], max_coords[1]]

        # Highlight the maximum value
        ax.scatter(max_z, max_y, color='red', s=5, label='Max Value')
        ax.text(
          max_z, max_y, f'{var[max_coords]:.4f}', 
          color='yellow', fontsize=11, fontweight=550,
          ha='left', va='bottom', 
          bbox=dict(facecolor='black', edgecolor='none', alpha=0.0)  # Set transparency with alpha
        )

    # Save the plot
    plt.savefig(savepath + pretext + fieldname + '.png')
    plt.close(fig)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")