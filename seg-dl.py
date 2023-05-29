import os
import numpy as np
import pandas as pd
import nibabel as nib
import re

from scipy.ndimage import zoom
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Lambda

import seaborn as sns
import matplotlib.pyplot as plt
from nilearn import plotting
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.image

import warnings
warnings.filterwarnings("ignore")

# define directory
input_folder = './data/vanderbilt/' # use ./data/small_batch_test/' for testing

# load demographics data
dem = pd.read_csv("./data/vanderbilt_ct_phenotype_2-14-23.csv") # use pd.read_excel("./CCF_CT_demographic.xlsx") for CCF

if not os.path.exists("./data/projections"): os.makedirs("./data/projections")
if not os.path.exists("./slices"): os.makedirs("./slices")
if not os.path.exists("./plots"): os.makedirs("./plots")

# define desired output voxel size
output_spacing = (1.0, 1.0, 1.0)  # 1 mm isotropic spacing

# define number of clusters to create
n_clusters = 2

# load all NIfTI files in input folder
resampled_data = []
projected_data = []
scan_IDs = []

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
        proj_path = "./data/projections/" + file_name[:-7] + ".png"
        matplotlib.image.imsave(proj_path, proj)

        projected_data.append(proj)

# extract the image_id and af_recur columns and store in a new dataframe
af_recur_status = dem[['study_id', 'recurrence']].astype({"study_id": "string"})

# filter the dataframe to only include rows where image_id is in the scan_id vector
af_recur_status = af_recur_status[af_recur_status['study_id'].isin(scan_IDs)]

# sort the dataframe based on the order of the scan_id vector
af_recur_status = af_recur_status.set_index('study_id').loc[scan_IDs].reset_index()

af_recur = np.array(af_recur_status['recurrence'].values).reshape(-1, 1)
preprocessed_images = np.array(projected_data).reshape(-1, 500, 500, 1)

X_train, X_test, y_train, y_test = train_test_split(projected_data, af_recur, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

X_train = np.array(X_train).reshape(-1, 500, 500, 1)
X_val = np.array(X_val).reshape(-1, 500, 500, 1)
X_test = np.array(X_test).reshape(-1, 500, 500, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(500, 500, 3))

# Freeze the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model
model = Sequential()
model.add(Lambda(lambda x: tf.image.grayscale_to_rgb(x)))  # Convert grayscale to RGB
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16)

# Predicting probabilities for the positive class in the testing set
y_test_prob = model.predict(X_test)

# Calculating false positive rate, true positive rate, and threshold for ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)

# Calculating the area under the ROC curve (AUC)
roc_auc = auc(fpr, tpr)

# Plotting the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()