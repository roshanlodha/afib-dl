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
ccf = pd.read_excel("/home/lodhar/afib-dl/data/CT_demographic_w_ccf_variables 20231102 hsk3.xlsx")
ccf['best_nii_dir'] = '/home/lodhar/afib-dl/nifti/ccf/' + ccf['image_id'].astype(str) + '.nii.gz'
ccf['exists'] = ccf.apply(lambda row: os.path.isfile(row['best_nii_dir']), axis = 1)
ccf['exists'].value_counts()

# drop samples with no associated scan
ccf_with_scans = ccf[ccf['exists']].drop(['exists'], axis = 1)
ccf_with_scans['af_recur'].value_counts()

# abnormal = recurrence (1) and normal = no recurrence (0)
abnormal_scan_paths = ccf_with_scans[ccf_with_scans['af_recur'] == 1]['best_nii_dir'].values
print("number of abnormal scans: " + str(len(abnormal_scan_paths)))

normal_scan_paths = ccf_with_scans[ccf_with_scans['af_recur'] == 0]['best_nii_dir'].values
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
    try:
        # Read scan
        volume = read_nifti_file(path)
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
print("reading abnormal scans...") # consider tqdm(random.sample(abnormal_scan_paths, n_scans))
abnormal_scans = np.array([process_scan(path) for path in abnormal_scan_paths], dtype = "object") 
print("filtering abnormal scans...")
abnormal_scans = np.array(list(filter(lambda item: item is not None, abnormal_scans)))

print("reading normal scans...") # consider tqdm(random.sample(normal_scan_paths, n_scans))
normal_scans = np.array([process_scan(path) for path in normal_scan_paths], dtype = "object")
print("filtering normal scans...")
normal_scans = np.array(list(filter(lambda item: item is not None, normal_scans)))

abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

x_test = np.concatenate((abnormal_scans, 
                         normal_scans), axis=0)
y_test = np.concatenate((abnormal_labels, 
                         normal_labels), axis=0)

# save data for downstream ML use
dataset_dict = {"x_test": x_test, "y_test": y_test}

with open('../test_data.pkl', 'wb') as file:
    pickle.dump(dataset_dict, file)