# OS
import os 
import sys
import pickle
from pathlib import Path

# math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import random

# .nii
import nibabel as nib
import nilearn as nil

vandy = pd.read_csv('/home/lodhar/afib-dl/data/vanderbilt_ct_phenotype_2-14-23.csv')

vandy['la_seg_dir'] = "/home/lodhar/afib-dl/segmentations/LA/Cardiac_" + vandy['study_id'].astype(str) + "_la_smooth_1_5_50.38.nii.gz"
vandy = vandy[vandy['la_seg_dir'].apply(lambda x: os.path.exists(x))]
vandy['la_seg_cropped_dir'] = "/home/lodhar/afib-dl/segmentations/LA/cropped/" + vandy['study_id'].astype(str) + ".nii.gz"

def get_nifti_shape(filepath):
    try:
        img = nib.load(filepath)
        return img.shape
    except:
        return None
    
vandy['shape'] = vandy['la_seg_dir'].apply(get_nifti_shape)
vandy[['x_dim', 'y_dim', 'z_dim']] = vandy['shape'].apply(lambda shape: pd.Series(shape) if shape is not None else pd.Series([None, None, None]))

min_dim_size = (vandy['x_dim'].min(), vandy['y_dim'].min(), vandy['z_dim'].min())
print("minimum dimensions: " + str(min_dim_size))

# abnormal = recurrence (1) and normal = no recurrence (0)
abnormal_scan_paths = vandy[vandy['recurrence'] == 1]['la_seg_dir'].values
print("number of abnormal scans: " + str(len(abnormal_scan_paths)))

normal_scan_paths = vandy[vandy['recurrence'] == 0]['la_seg_dir'].values
print("number of normal scans: " + str(len(normal_scan_paths)))

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndi.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndi.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def center_crop(image, crop_dims):
    image_dims = image.shape
    crop_starts = [(image_dims[i] - crop_dims[i]) // 2 for i in range(3)]
    crop_ends = [crop_starts[i] + crop_dims[i] for i in range(3)]
    cropped_image = image[crop_starts[0]:crop_ends[0], crop_starts[1]:crop_ends[1], crop_starts[2]:crop_ends[2]]
    return cropped_image

def process_scan(path, new_dims):
    """Read and resize volume"""
    try:
        # Read scan
        volume = read_nifti_file(path)
        # Center crop
        volume = center_crop(volume, new_dims)
        # Normalize
        volume = normalize(volume)
        # Resize width, height and depth
        volume = resize_volume(volume)
        if volume is None:
            print(path + " volume is None")
            raise Exception()
        else:
            print(path + " read successfully")
            return volume
    except:
        print(path + " could not be read")

# process scans
print("reading abnormal scans...")
abnormal_scans = np.array([process_scan(path, min_dim_size) for path in abnormal_scan_paths], dtype = "object") 
non_filtered_abnormal_paths = [path for path, scan in zip(abnormal_scan_paths, abnormal_scans) if scan is not None]
print("filtering abnormal scans...")
abnormal_scans = np.array(list(filter(lambda item: item is not None, abnormal_scans)))

print("reading normal scans...")
normal_scans = np.array([process_scan(path, min_dim_size) for path in normal_scan_paths], dtype = "object")
non_filtered_normal_paths = [path for path, scan in zip(normal_scan_paths, normal_scans) if scan is not None]
print("filtering normal scans...")
normal_scans = np.array(list(filter(lambda item: item is not None, normal_scans)))

abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# 70/30 test-train split (testing set will be biased towards no recurrence d.t. differences in sample size)
n_scans = min(len(abnormal_scan_paths), len(normal_scan_paths))
split_point = int(0.7 * n_scans)

x_train = np.concatenate((abnormal_scans[:split_point], 
                          normal_scans[:split_point]), axis=0)
y_train = np.concatenate((abnormal_labels[:split_point], 
                          normal_labels[:split_point]), axis=0)
x_val = np.concatenate((abnormal_scans[split_point:n_scans], 
                        normal_scans[split_point:n_scans]), axis=0)
y_val = np.concatenate((abnormal_labels[split_point:n_scans], 
                        normal_labels[split_point:n_scans]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)


training_dir = non_filtered_abnormal_paths[:split_point] + non_filtered_normal_paths[:split_point]
validataion_dir = non_filtered_abnormal_paths[split_point:n_scans] + non_filtered_normal_paths[split_point:n_scans]

print("writing training paths...")
with open("/home/lodhar/afib-dl/data/segmentations/vandy_training_dir.txt", "w") as file:
    for path in training_dir:
        file.write(path + "\n")

print("writing validations paths...")
with open("/home/lodhar/afib-dl/data/segmentations/vandy_validation_dir.txt", "w") as file:
    for path in validataion_dir:
        file.write(path + "\n")

print("writing training and validations processed scans...")
dataset_dict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}

with open('/home/lodhar/afib-dl/data/segmentations/processed/LA/train.pkl', 'wb') as file:
    pickle.dump(dataset_dict, file)