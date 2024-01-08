import os 
import sys
import pickle
from pathlib import Path

import numpy as np
import random
import scipy.ndimage as ndi
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import nilearn as nil

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras.utils import plot_model
from keras import layers

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

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndi.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume

def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu", name = "intermediate")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

def one_hot_encode(original_dataframe, feature_to_encode):
    original_dataframe[feature_to_encode] = original_dataframe[feature_to_encode].astype(str)
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 

vandy = pd.read_csv("./data/vanderbilt_ct_phenotype_2-14-23.csv")
temp = vandy.groupby('recurrence')
vandy = vandy.apply(lambda x: x.sample(gaf.size().min()).reset_index(drop=True))

vandy = vandy[vandy['la_any_modality'] != "."] # handling special "." characters
outcomes = [col for col in vandy.columns if 'recur' in col.lower()]
valve_dx_cols = [col for col in vandy.columns if 'type_valve_dx' in col.lower()]
ablation_cols = [col for col in vandy.columns if 'ablation' in col.lower()]

X = af.drop(outcomes + valve_dx_cols + ablation_cols, axis = 1).drop(['mri_ct'], axis = 1)
y = af['recurrence'].reset_index(drop=True)

features_to_binarize = X.select_dtypes('int64').columns
features_to_one_hot_encode = ['race', 'ethnicity']
for feature in features_to_one_hot_encode:
    X = one_hot_encode(X, feature)

# Build model.
model = get_model(width=128, height=128, depth=64)
model.load_weights("best_classifier.h5")

model = Model(input = model.input, output = model.get_layer("intermediate").output)

dlo = np.empty([X.shape[0], 512])

for i, row in X.iterrows():
    path = 'nifti/vandy/' + row['study_id'].astype(str) + '.nii.gz'
    try:
        dlo[i] = model.predict(np.expand_dims(process_scan(path), axis=0))
    except Exception as e:
        dlo[i] = np.repeat(np.nan, 512)

X_train, X_test, y_train, y_test = train_test_split(X.drop(['study_id'], axis = 1), y, test_size = 0.2) # this analysis keeps outliers as they are clinically meaningful
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import RocCurveDisplay

knn = KNeighborsClassifier()
param_grid = dict(n_neighbors=list(range(1, X_test.shape[0])))
grid = GridSearchCV(knn, param_grid, cv=10, scoring = 'roc_auc', return_train_score=False)
knn = grid.fit(X_train, y_train)
knn.best_params_

RocCurveDisplay.from_estimator(knn.best_estimator_, X_test, y_test)
plt.savefig("./figs/clinicalROC.png")

import pickle as pkl
# save the model to disk
filename = 'knn_clinical.sav'
pkl.dump(knn.best_estimator_, open(filename, 'wb'))
