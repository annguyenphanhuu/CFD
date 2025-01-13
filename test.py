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
# load data
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
    plt.savefig('./model_evaluation/' + pretext + fieldname + '.png')
    plt.close(fig)
    
# generate samples and save as a plot and save the model
def model_prediction(step, g_model, x_test_1, y_test_2, test_size, field):
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

    vmin = max(y_error.min(), Y.min(), y.min())
    vmax = max(y_error.max(), Y.max(), y.max())
    plot_image(X, str(step + 1) + '_Boundary_', field, 1, vmin=X.min(), vmax=X.max())
    plot_image(Y, str(step + 1) + '_CFD_', field, 3, vmin=vmin, vmax=vmax)
    plot_image(y, str(step + 1) + '_Predict_', field, 3, vmin=vmin, vmax=vmax)

    # Calculate and plot the error
    plot_image(y_error, str(step + 1) + '_error_abs_', field, 3, vmin=vmin, vmax=vmax)

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
x_test_1 = bnd_array[-216:]
y_train_1 = sflow_P_array[:1000]
y_test_1 = sflow_P_array[-216:]
y_train_2 = sflow_U_array[:1000]
y_test_2 = sflow_U_array[-216:]

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
  writer.writerow(['Test ID', 'Error max (%)'])
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

    
    ymax = Y.max()
    ymean = np.mean(y_test_1[ix])
    print('ymax', ymax, ' at test ID: ',str(ix))
    sum_err = 0
    for i in range(64):
      for j in range(512):
        # y_error[i,j] = abs(Y[i,j] - y[i,j])*100/Y_max # global
        y_error[i,j] = abs(Y[i,j] - y[i,j])
        sum_err += y_error[i,j]

    err_avg = sum_err/(64*512)
    print('Error average: ',err_avg)
    sum_error += err_avg

    plot_image(y_error,str(ix)+'_error_abs_', field, 3, vmin=vmin, vmax=vmax)


    
    

    
    # y_error = y_error*100/ymax
    Error_mean = np.asarray(y_error.max(axis=0))
    Error_max = Error_mean.max()
    print('Error max:',Error_max)
    writer.writerow([str(ix),str(Error_max)])

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