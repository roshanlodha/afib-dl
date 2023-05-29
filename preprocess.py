import os
import numpy as np
import pandas as pd
import nibabel as nib
import re

from scipy.ndimage import zoom

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.image

import warnings
warnings.filterwarnings("ignore")

# define directory
input_folder = './data/small_batch_test/' # use ./data/vanderbilt/' for training

# load demographics data
dem = pd.read_csv("./data/vanderbilt_ct_phenotype_2-14-23.csv") # use pd.read_excel("./CCF_CT_demographic.xlsx") for CCF

if not os.path.exists("./data/projections"): os.makedirs("./data/projections")
if not os.path.exists("./slices"): os.makedirs("./slices")
if not os.path.exists("./plots"): os.makedirs("./plots")

# define desired output voxel size
output_spacing = (1.0, 1.0, 1.0)  # 1 mm isotropic spacing

# load all NIfTI files in input folder
resampled_data = []
projected_data = []
scan_IDs = []

def crop_image(image):
    # Get the indices of non-zero elements along rows and columns
    row_indices = np.where(np.any(image != 0, axis=1))[0]
    col_indices = np.where(np.any(image != 0, axis=0))[0]

    # Determine the start and end indices for cropping
    start_row = row_indices[0]
    end_row = row_indices[-1] + 1
    start_col = col_indices[0]
    end_col = col_indices[-1] + 1

    # Calculate the dimensions of the cropped image
    width = end_col - start_col
    height = end_row - start_row

    # Determine the size of the square crop
    size = max(width, height)

    # Calculate the padding required to make the crop square
    pad_width = size - width
    pad_height = size - height

    # Calculate the padding amounts for top, bottom, left, and right
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Perform the cropping and padding
    cropped_image = image[start_row:end_row, start_col:end_col]
    padded_image = np.pad(cropped_image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    return padded_image

def pad_images(images):
    max_size = max(image.shape[0] for image in images)  # Maximum size among all images

    padded_images = []
    for image in images:
        # Calculate the padding amounts for top, bottom, left, and right
        pad_height = max_size - image.shape[0]
        pad_width = max_size - image.shape[1]
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Perform padding
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        padded_images.append(padded_image)

    return np.array(padded_images)

print("Loading and resampling NIfTI files...")
for file_name in tqdm(os.listdir(input_folder), desc='Progress', unit='image'):
    if file_name.endswith('.nii.gz'):
        if not int(re.search(r"Cardiac_(\d+)_", file_name).group(1)) in dem['study_id'].unique():
            pass
        file_path = os.path.join(input_folder, file_name)
        img = nib.load(file_path)
        scan_IDs.append(re.search(r"Cardiac_(\d+)_", file_name).group(1))

        # resample the loaded image
        input_spacing = img.header.get_zooms()
        resampling_factors = tuple(np.array(input_spacing) / np.array(output_spacing))
        resampled_image = zoom(img.get_fdata(), resampling_factors, order=1)

        # pad the images to equal size
        target_shape = (500, 500, 500)
        x_pad = max(target_shape[0] - resampled_image.shape[0], 0)
        y_pad = max(target_shape[1] - resampled_image.shape[1], 0)
        z_pad = max(target_shape[2] - resampled_image.shape[2], 0)
        x_pad_before = x_pad // 2
        x_pad_after = x_pad - x_pad_before
        y_pad_before = y_pad // 2
        y_pad_after = y_pad - y_pad_before
        z_pad_before = z_pad // 2
        z_pad_after = z_pad - z_pad_before
        resampled_image_padded = np.pad(resampled_image, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (z_pad_before, z_pad_after)), mode='constant', constant_values=0)

        resampled_data.append(resampled_image_padded)

        # create a 2D projection of the 3D images
        proj = np.sum(resampled_image_padded, axis = 2)
        proj = crop_image(proj)
        proj_path = "./data/projections/" + file_name[:-7] + ".png"
        matplotlib.image.imsave(proj_path, proj)

        projected_data.append(proj)

padded_data = pad_images(projected_data)

# extract the image_id and af_recur columns and store in a new dataframe
af_recur_status = dem[['study_id', 'recurrence']].astype({"study_id": "string"})

# filter the dataframe to only include rows where image_id is in the scan_id vector
af_recur_status = af_recur_status[af_recur_status['study_id'].isin(scan_IDs)]

# sort the dataframe based on the order of the scan_id vector
af_recur_status = af_recur_status.set_index('study_id').loc[scan_IDs].reset_index()

af_recur = np.array(af_recur_status['recurrence'].values).reshape(-1, 1)
preprocessed_images = np.array(projected_data).reshape(-1, 500, 500, 1)

af_recur.tofile('./cache/af_recur.dat')
preprocessed_images.tofile('./cache/projected_data.dat')
resampled_data.tofile('./cache/resampled_data.dat')