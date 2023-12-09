# OS 
import os 
import sys
import pickle
from pathlib import Path
from tqdm import tqdm

# Math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import random

# .nii
import nibabel as nib
import nilearn as nil

# load samples
vandy = pd.read_csv("./data/vanderbilt_ct_phenotype_2-14-23.csv")
vandy['best_nii_dir'] = 'nifti/vandy/' + vandy['study_id'].astype(str) + '.nii.gz'
vandy['exists'] = vandy.apply(lambda row: os.path.isfile(row['best_nii_dir']), axis = 1)
vandy['exists'].value_counts()

# drop samples with no associated scan
vandy_with_scans = vandy[vandy['exists']].drop(['exists'], axis = 1)
vandy_with_scans['recurrence'].value_counts()

# abnormal = recurrence (1) and normal = no recurrence (0)
abnormal_scan_paths = vandy_with_scans[vandy_with_scans['recurrence'] == 1]['best_nii_dir'].values
print("number of abnormal scans: " + str(len(abnormal_scan_paths)))

normal_scan_paths = vandy_with_scans[vandy_with_scans['recurrence'] == 0]['best_nii_dir'].values
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

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume

# process scans
abnormal_scans = np.array([process_scan(path) for path in tqdm(abnormal_scan_paths)]) # consider random.sample(abnormal_scan_paths, n_scans)
normal_scans = np.array([process_scan(path) for path in tqdm(normal_scan_paths)]) # consider random.sample(normal_scan_paths, n_scans)

abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# 70/30 test-train split (testing set will be biased towards no recurrence d.t. differences in sample size)
n_scans = min(len(abnormal_scan_paths), len(normal_scan_paths))
split_point = int(0.7 * n_scans)

x_train = np.concatenate((abnormal_scans[:split_point], 
                          normal_scans[:split_point]), axis=0)
y_train = np.concatenate((abnormal_labels[:split_point], 
                          normal_labels[:split_point]), axis=0)
x_val = np.concatenate((abnormal_scans[split_point:], 
                        normal_scans[split_point:]), axis=0)
y_val = np.concatenate((abnormal_labels[split_point:], 
                        normal_labels[split_point:]), axis=0)
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)

# save data for downstream ML use
dataset_dict = {"x_train": x_train, "x_val": x_val, "y_train": y_train, "y_val": y_val}

with open('data.pkl', 'wb') as file:
    pickle.dump(dataset_dict, file)