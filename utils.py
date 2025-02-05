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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False


# select a batch of random samples, returns images and target
def generate_real_samples(x_train_1, y_train_2, ix, patch_shape):
    # retrieve selected images
    X1 = []
    X2 = []
    cons_arr = []
    for i in ix:
        x1 = x_train_1[i]
        X1.append(x1)
        x2 = y_train_2[i]
        X2.append(x2)
        # cons = cond_array[i]
        # cons_arr.append(cons)
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
		x = g_model.predict([X_realA[[i]]])
		x = np.asarray(x[0])
		X.append(x)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape * 8, 1))
	X = np.asarray(X).astype('float32')
	return X, y

def load_data(processed_folder):
    bnd_array = []
    sflow_P_array = []
    sflow_U_array = []
    DeltaP_array = []
    cond_array = [] 

    for file_path in os.listdir(processed_folder):
        with open(os.path.join(processed_folder, file_path), 'r') as filename:
            file = csv.DictReader(filename)
            P_array = []
            U_array = []
            pts = np.ones((64,512))
            p = np.zeros((64,512))
            u = np.zeros((64,512))

            # Extract data from CSV columns
            for row in file:
                P_array.append(float(row['p_convert']))
                U_array.append(float(row['Umean_center']))

            # Extract constriction value (first row's value)
            # cond_array.append([float(row['cond1']), float(row['cond2']), float(row['cond3'])])

            # Map data into the domain (4x50 mm)
            for i in range(64):
                for j in range(512):
                    p[i, j] = P_array[i * 512 + j]
                    u[i, j] = U_array[i * 512 + j]
                    pts[i, j] = -1 if P_array[i * 512 + j] > 0 else 1

            sflow_P_array.append(p)
            sflow_U_array.append(u)
            DeltaP_array.append(np.mean(p[:, 0]) - np.mean(p[:, -1]))

            # Compute SDF (Signed Distance Function)
            phi = pts
            d_x = 4 / 64
            d = skfmm.distance(phi, d_x)
            bnd_array.append(np.array(d))

    # Ensure there is at least one constriction value to return

    return bnd_array, sflow_P_array, sflow_U_array, DeltaP_array

def plot_image(savepath, var, pretext, fieldname, flag, vmin=None, vmax=None, rotate_contour=False, delta_p=0):
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
    elif flag == 5:
        labeltxt = 'Delta Pressure (Pa)'

    var[np.isinf(var)] = np.nan
    var = np.clip(var, vmin, vmax)

    Z, Y = np.meshgrid(np.linspace(0, 50, var.shape[1]), np.linspace(0, 4, var.shape[0]))

    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    contour = ax.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow, vmin=vmin, vmax=vmax)

    if "Delta_Pressure" in pretext.lower():
        ax.text(
            0, 0, f'{delta_p[0]:.4f}, {delta_p[1]:.4f}', 
            color='yellow', fontsize=11, fontweight=550,
            ha='left', va='bottom', 
            bbox=dict(facecolor='black', edgecolor='none', alpha=0.0)  # Set transparency with alpha
        )
    if rotate_contour: 
        fig.colorbar(contour, ax=ax, label=labeltxt,orientation='horizontal')
    else:
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


# generate samples and save as a plot and save the model
def summarize_performance(directory, step, flag, g_model, x_test_1, y_test_2, test_size, field, scale=1, p=None):
    # Generate a batch of fake samples
    ix = random.randint(0, test_size - 1)
    X_fakeB, _ = generate_fake_samples(g_model, x_test_1[[ix]], 1)

    X = x_test_1[ix]
    X = np.squeeze(X, axis=2)

    y = X_fakeB[0] * scale
    
    Y = y_test_2[ix] * scale
    
    if flag != 5:
        y = np.squeeze(y, axis=2)
        Y = np.squeeze(Y, axis=2)

        y_error = abs(Y - y)

        vmin = min(y_error.min(), Y.min(), y.min())
        vmax = max(y_error.max(), Y.max(), y.max())

        print('X shape', X.shape)
        print('Y shape', Y.shape)
        print('y shape', y.shape)

        # Plot each variable with its own min and max
        plot_image(directory, X, str(step + 1) + '_Boundary_', field, 1, vmin=X.min(), vmax=X.max())
        plot_image(directory, Y, str(step + 1) + '_CFD_', field, flag, vmin=vmin, vmax=vmax)
        plot_image(directory, y, str(step + 1) + '_Predict_', field, flag, vmin=vmin, vmax=vmax)

        # Calculate and plot the error
        plot_image(directory, y_error, str(step + 1) + '_error_abs_', field, flag, vmin=vmin, vmax=vmax)
    else:
        plot_image(directory, X, str(step + 1) + '_Boundary_', field, 1, vmin=X.min(), vmax=X.max())
        P = p[ix]
        y_error = abs(Y - y)
        plot_image(directory, P, str(step + 1) + '_Delta_Pressure_', field, 2, vmin=P.min(), vmax=P.max(), delta_p=y_error[0])

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")