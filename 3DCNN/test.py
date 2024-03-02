import pickle

import numpy as np
import random
import scipy.ndimage as ndi
from sklearn import metrics

import tensorflow as tf
import keras
from keras.utils import plot_model
from keras import layers

import matplotlib.pyplot as plt

with open('/home/lodhar/afib-dl/data/test_data.pkl', 'rb') as file:
    dataset_dict = pickle.load(file)

x_test = np.asarray(dataset_dict['x_test']).astype('float32')

y_test = np.asarray(dataset_dict['y_test']).astype('float32')

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

# Define data loaders.
validation_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
batch_size = 1

# Only rescale.
validation_dataset = (
    validation_loader.shuffle(len(x_test))
    .map(validation_preprocessing)
    .batch(batch_size)
    .prefetch(2)
)

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

model.load_weights("/home/lodhar/afib-dl/3DCNN/best_classifier.h5")

y_pred = model.predict(x_test).ravel()
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
best_auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, linestyle='--', label='No Skill')
plt.title("Best Model AUC")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.text(0.6, 0, 'AUC = %s' %(best_auc))
plt.savefig("../figs/testAUC.png")