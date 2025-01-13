import zipfile
import csv
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import cv2
import matplotlib.pyplot as plt
import skfmm
import argparse
import shutil

def extract_zip(zip_folder_path, extract_to):
    try:
        with zipfile.ZipFile(zip_folder_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                try:
                    zip_ref.extract(file, extract_to)
                    #print(f"Unzip: {file}")
                except Exception as e:
                    print(f"Skip error filefile {file}: {e}")
    except zipfile.BadZipFile:
        print(f"Error ZIP {zip_folder_path}")
    except Exception as e:
        print(f"Catch Error: {e}")

def process_zip_folder(zip_folder_path, output_folder):
    # Extract the initial zip folder containing multiple zip files
    extract_to = Path(output_folder) / 'unzipped'
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_folder_path} to {extract_to}")
    extract_zip(zip_folder_path, extract_to)

    # Unzip individual files
    for zip_file in extract_to.glob('*.zip'):
        folder_name = zip_file.stem
        folder_extract_path = extract_to / folder_name
        folder_extract_path.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_file} to {folder_extract_path}")
        extract_zip(zip_file, folder_extract_path)
        try:
            zip_file.unlink()
        except Exception as e:
            print(f"Error deleting {zip_file}: {e}")
        # Locate the nested folder containing midBox.csv
        nested_folders = folder_extract_path / folder_name / "postProcessing" / "boxUniform"
        if nested_folders.exists():
            #print(f"Found boxUniform in {nested_folders}")
            number_folders = [(float(d.name), d) for d in nested_folders.iterdir() if d.is_dir() and d.name.replace('.', '', 1).isdigit()]
            if number_folders:
                largest_folder = max(number_folders, key=lambda x: x[0])[1]
                #print(f"Largest folder: {largest_folder}")
                mid_box_file = largest_folder / 'midBox.csv'
                if mid_box_file.exists():
                    print(f"Found midBox.csv in {largest_folder}")
                    df = pd.read_csv(mid_box_file)
                    output_csv_path = output_folder / f"{folder_name}_midBox.csv"
                    df.to_csv(output_csv_path, index=False)
                else:
                    print(f"midBox.csv not found in {largest_folder}")
            else:
                print(f"No valid numeric folders found in {nested_folders}")
        else:
            print(f"boxUniform folder not found in {folder_extract_path / folder_name / 'postProcessing'}")

    # Delete the 'unzipped' folder after processing
    print(f"Deleting the unzipped folder: {extract_to}")
    shutil.rmtree(extract_to)

def calculate_umean_and_restructure(input_csv_path, output_csv_path):
    # Read the input CSV file
    df = pd.read_csv(input_csv_path)
    # Step 2: Calculate Umean = sqrt(U_x^2 + U_y^2 + U_z^2)
    df["Umean"] = np.sqrt(df["U_x"]**2 + df["U_y"]**2 + df["U_z"]**2)
    # Step 3: Drop columns "U_x", "U_y", "U_z"
    df.drop(columns=["U_x", "U_y", "U_z"], inplace=True)
    # Step 4: Keep only the columns "x", "y", "z", "p", and "Umean"
    df = df[["x", "y", "z", "p", "Umean"]]
    # Step 5: Save the restructured file to the output path
    df.to_csv(output_csv_path, index=False)
    print(f"Calculate Umean: {output_csv_path}")

def process_all_csvs_in_output_folder(output_folder):
    processed_folder = Path(output_folder) / "processed"
    processed_folder.mkdir(parents=True, exist_ok=True)
    for csv_file in Path(output_folder).glob("*.csv"):
        output_csv_path = processed_folder / csv_file.name
        calculate_umean_and_restructure(csv_file, output_csv_path)

def remove_duplicates_and_save_in_place(csv_path):
    df = pd.read_csv(csv_path)
    # Remove duplicate rows
    df = df.drop_duplicates()
    df.to_csv(csv_path, index=False)
    print(f"Remove duplicates: {csv_path}")

def remove_duplicates_in_folder(folder_path):
    for csv_file in Path(folder_path).glob("*.csv"):
        remove_duplicates_and_save_in_place(csv_file)

def process_csv_with_pixel_coordinates_nearest(input_csv, output_csv):
    """
    Process a CSV file using nearest interpolation and compute pixel coordinates.
    """
    # Physical size of the domain
    y_range_physical = 4e-3  # 4mm
    z_range_physical = 50e-3  # 50mm

    # Desired pixel dimensions
    ny_final, nz_final = 40, 500

    # Pixel sizes
    dy_final, dz_final = y_range_physical / ny_final, z_range_physical / nz_final   # 0.0001m/pixel

    # Pixel centers (y and z)
    y_final = -0.002 + (np.arange(ny_final) + 0.5) * dy_final   # y from -2mm to +2mm
    z_final = (np.arange(nz_final) + 0.5) * dz_final    # z from 0mm to 50mm

    # Mesh grid for final coordinates
    Y_grid, Z_grid = np.meshgrid(y_final, z_final, indexing='ij')

    # Read input data
    data = pd.read_csv(input_csv)
    arr = data.to_numpy()

    # Assuming structure: x, y, z, p, Umean
    Y_all = arr[:, 1]  # y coordinates (meters)
    Z_all = arr[:, 2]  # z coordinates (meters)
    P_all = arr[:, 3]  # pressure
    U_all = arr[:, 4]  # velocity magnitude
    points = np.column_stack((Y_all, Z_all))

    # Determine y_min(z), y_max(z) from original data
    z_unique_samples = np.linspace(0, z_range_physical, 101)  # 101 slices

    y_min_list, y_max_list = [], []
    for z_val in z_unique_samples:
        idx_z = np.where(np.abs(Z_all - z_val) < 1e-5)[0]
        if len(idx_z) > 0:
            y_min_list.append(np.min(Y_all[idx_z]))
            y_max_list.append(np.max(Y_all[idx_z]))
        else:
            idx_z_near = np.where(np.abs(Z_all - z_val) < 1e-4)[0]
            if len(idx_z_near) > 0:
                y_min_list.append(np.min(Y_all[idx_z_near]))
                y_max_list.append(np.max(Y_all[idx_z_near]))
            else:
                y_min_list.append(np.nan)
                y_max_list.append(np.nan)

    y_min_list, y_max_list = np.array(y_min_list), np.array(y_max_list)

    # Interpolate y_min(z) and y_max(z) for all z_final
    valid_mask = ~np.isnan(y_min_list) & ~np.isnan(y_max_list)
    z_valid, y_min_valid, y_max_valid = z_unique_samples[valid_mask], y_min_list[valid_mask], y_max_list[valid_mask]
    y_min_interp = np.interp(z_final, z_valid, y_min_valid)
    y_max_interp = np.interp(z_final, z_valid, y_max_valid)

    # Interpolate P and U onto the final grid using nearest interpolation
    P_final = griddata(points, P_all, (Y_grid, Z_grid), method='nearest', fill_value=0)
    U_final = griddata(points, U_all, (Y_grid, Z_grid), method='nearest', fill_value=0)

    # Apply geometry mask
    for j in range(nz_final):
        y_min_j, y_max_j = y_min_interp[j], y_max_interp[j]
        outside_mask = (y_final < y_min_j) | (y_final > y_max_j)
        P_final[outside_mask, j], U_final[outside_mask, j] = 0.0, 0.0

    output_data = [[y_final[i], z_final[j], P_final[i, j], U_final[i, j]] for i in range(ny_final) for j in range(nz_final)]
    df_out = pd.DataFrame(output_data, columns=['y_center', 'z_center', 'p_center', 'Umean_center'])
    df_out.to_csv(output_csv, index=False)
    print(f"Nearest interpolation: {output_csv}")

def process_all_csvs_with_pixel_coordinates(output_folder):
    for csv_file in Path(output_folder).glob("*.csv"):
        process_csv_with_pixel_coordinates_nearest(csv_file, csv_file)

def plot_image(var, pretext, fieldname, flag, output_dir):
    """
    Generates and saves images for the given data array.
    """
    if flag == 1:
        labeltxt = 'SDF Boundary'
    elif flag == 2:
        labeltxt = 'Pressure (Pa)'
    elif flag == 3:
        labeltxt = 'U mean (m/s)'

    Z, Y = np.meshgrid(np.linspace(0, 50, 500), np.linspace(0, 4, 40))
    fig = plt.figure()

    plt.gca().set_aspect('equal', adjustable='box')
    plt.contourf(Z, Y, var, 50, cmap=plt.cm.rainbow)
    plt.colorbar(label=labeltxt)

    # Save the plot to the specific folder for this CSV file
    plt.savefig(output_dir / f"{pretext}{fieldname}.png")
    plt.close('all')

def generate_sdf_visualization(phi, output_dir, case_name):
    """
    Generates and saves Signed Distance Function (SDF) plots.
    """
    d_x = 4 / 40  # Grid spacing for SDF computation
    d = skfmm.distance(phi, d_x)

    # Create Z and Y meshgrid
    Z, Y = np.meshgrid(np.linspace(0, 50, 500), np.linspace(0, 4, 40))

    # First plot: imshow of SDF
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(d, vmin=d.min(), vmax=d.max(), origin='lower', extent=[Z.min(), Z.max(), Y.min(), Y.max()])
    plt.savefig(output_dir / f"{case_name}_SDF_imshow.png")
    plt.close()

    # Second plot: contour of SDF
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal', adjustable='box')
    CS = ax.contour(Z, Y, d, 15)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('SDF Value')
    plt.savefig(output_dir / f"{case_name}_SDF_contour.png")
    plt.close()

def process_csv(file_path, output_dir):
    """
    Processes a single CSV file, calculates SDF, and generates CFD and SDF visualizations.
    """
    print(f"Processing file: {file_path.name}")

    # Create a folder for this CSV file's output images
    csv_name = file_path.stem  # Get the file name without extension
    file_output_dir = output_dir / csv_name
    file_output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Read the CSV file
    with open(file_path, 'r') as filename:
        file = csv.DictReader(filename)
        P_array = []
        U_array = []
        pts = np.ones((40, 500))
        p = np.zeros((40, 500))
        u = np.zeros((40, 500))

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
        phi = pts
        d_x = 4 / 40
        d = skfmm.distance(phi, d_x)

        # Create output visualizations
        bnd = np.array(d)
        P = np.array(p)
        U = np.array(u)

        # Plot SDF, Pressure, and Umean visualizations to the respective folder
        plot_image(bnd, "Test_SDF_", "Boundary", 1, file_output_dir)
        plot_image(P, "Test_CFD_", "Pressure", 2, file_output_dir)
        plot_image(U, "Test_CFD_", "Umean", 3, file_output_dir)

        # Generate additional SDF visualizations
        generate_sdf_visualization(phi, file_output_dir, csv_name)

# Apply to all CSV files in the processed folder
def process_all_csvs_with_plots(output_folder):
    processed_folder = Path(output_folder) / "processed"
    for csv_file in Path(processed_folder).glob("*.csv"):
        process_csv(csv_file, Path(output_folder))


def main(zip_path, output_folder):
    # Create output folder if not exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Extract files, process CSVs, remove duplicates, and apply pixel coordinate processing
    process_zip_folder(zip_path, output_folder)
    process_all_csvs_in_output_folder(output_folder)
    processed_folder = Path(output_folder) / 'processed'
    remove_duplicates_in_folder(processed_folder)
    process_all_csvs_with_pixel_coordinates(processed_folder)
    # Call this function after removing duplicates and restructuring CSVs
    process_all_csvs_with_plots(output_folder)

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Process and extract data from a zip file containing CSVs.')
    
    # Add arguments with default values
    parser.add_argument('--zip_path', type=str, nargs='?', default='./Case_10.zip', help='Path to the zip folder (default: ./Case_15.zip).')
    parser.add_argument('--output_folder', type=str, nargs='?', default='./output_folder', help='Directory where extracted and processed files will be saved (default: ./output_folder).')

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with arguments
    main(args.zip_path, args.output_folder)
