#!/usr/bin/env python
# coding: utf-8

# # 3D image classification from CT scans

# ## Setup

# In[1]:


import os
import zipfile
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


# ## Preprocessing Samples
# CT scans are read in the Nifti format (extension .nii) using the `nibabel` package. CT scans store raw voxel intensity in Hounsfield units (HU) which range from -1024 to above 2000 in this dataset. Regions with an intensity above 400 are bones, so a threshold between -1000 and 400 is commonly used to normalize CT scans.
# 
# To process the data, we do the following:
# 
# We first rotate the volumes by 90 degrees, so the orientation is fixed
# We scale the HU values to be between 0 and 1.
# We resize width, height and depth.
# 
# Here we define several helper functions to process the data. These functions will be used when building training and validation datasets.

# In[2]:


import nibabel as nib

from scipy import ndimage


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
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    print("processed " + path)
    return volume


# ## Building train and validation datasets

# In[3]:


import pandas as pd

ccf_image_ids = [f.removesuffix(".nii.gz") for f in os.listdir(os.path.join(os.getcwd(), "data", "ccf/pulm"))]

ccf_labels_df = pd.read_excel("./data/CCF_CT_demographic.xlsx") # read

# filter demographic table based on available scans
ccf_labels_df['image_id'] = ccf_labels_df['image_id'].astype(str)
ccf_labels_df = ccf_labels_df[ccf_labels_df['image_id'].isin(ccf_image_ids)]

# sort the demographics table based on the scan order
ccf_labels_df['image_id'] = ccf_labels_df['image_id'].astype("category")
ccf_labels_df['image_id'] = ccf_labels_df['image_id'].cat.set_categories(ccf_image_ids)


# In[4]:


ccf_labels = ccf_labels_df['af_recur']

mask = np.array(ccf_labels_df['af_recur'].isna())
ccf_labels = np.ma.masked_array(ccf_labels, mask).compressed().astype(int)

ccf_scan_paths = [
    os.path.join(os.getcwd(), "data", "ccf/pulm", x)
    for x in os.listdir(os.path.join(os.getcwd(), "data", "ccf/pulm"))
]
ccf_scan_paths = np.ma.masked_array(ccf_scan_paths, mask).compressed()

n_train = int(len(ccf_labels)*0.8) if len(ccf_labels) == len(ccf_scan_paths) else None
print("number of training samples: " + str(n_train))


# In[ ]:


# Read and process the scans.
# Each scan is resized across height, width, and depth and rescaled.
ccf_scans = np.array([process_scan(path) for path in ccf_scan_paths])

# Split data in the ratio 80-20 for training and validation.
x_train = ccf_scans[:n_train]
y_train = ccf_labels[:n_train]
x_val = ccf_scans[n_train:]
y_val = ccf_labels[n_train:]
print(
    "Number of samples in train and validation are %d and %d."
    % (x_train.shape[0], x_val.shape[0])
)


# ## Data Augmentation
# 
# Data augmentation is done here by rotating the CT scans at random angles during training. This is used to artificially expand the data-set. This is helpful when we are given a data-set with very few data samples as in our data

# In[ ]:


import random

from scipy import ndimage


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
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


# While defining the train and validation data loader, the training data is passed through and augmentation function which randomly rotates volume at different angles. Note that the validation data is not rotated.

# In[ ]:


# Define data loaders.
train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

batch_size = 2
# Augment the on the fly during training.
train_dataset = (
    train_loader.shuffle(len(x_train))
    .map(train_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)
# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_val))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)


# ### Visualizing CT scans

# In[ ]:


import matplotlib.pyplot as plt

data = train_dataset.take(1)
images, labels = list(data)[0]
images = images.numpy()
image = images[0]
print("Dimension of the CT scan is:", image.shape)
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")


# In[ ]:


def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


# Visualize montage of slices.
# 4 rows and 10 columns for 100 slices of the CT scan.
plot_slices(4, 10, 128, 128, image[:, :, :40])


# ## Define a 3D convolutional neural network
# The architecture of the 3D CNN used in this example
# is based on [this paper](https://arxiv.org/abs/2007.13224).

# In[ ]:


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
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()


# ## Train model

# In[ ]:


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3d_image_classification.h5", save_best_only=True
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

# Train the model, doing validation at the end of each epoch
epochs = 100
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    shuffle=True,
    verbose=2,
    callbacks=[checkpoint_cb, early_stopping_cb],
)


# ## Visualizing model performance

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20, 3))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


# ## Made predictions on a single CT scan

# In[ ]:


model.load_weights("3d_image_classification.h5")
prediction = model.predict(np.expand_dims(x_val[0], axis=0))[0]
scores = [1 - prediction[0], prediction[0]]

class_names = ["normal", "abnormal"]
for score, name in zip(scores, class_names):
    print(
        "This model is %.2f percent confident that CT scan is %s"
        % ((100 * score), name)
    )

