import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from keras import layers
import nibabel as nib
import scipy.ndimage as ndi

# Define your model architecture and load weights
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
intermediate_layer_model = keras.Model(inputs=model.input, outputs=model.layers[-2].output)

model.load_weights("/home/lodhar/afib-dl/3DCNN/best_classifier.h5")

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

# Define a function to make predictions on a single CT scan
def predict_single_scan(scan_path, model = model):
    # Process the CT scan
    processed_scan = process_scan(scan_path)
    if processed_scan is None:
        return None  # Handle error if processing fails
    
    # Add batch dimension
    processed_scan = np.expand_dims(processed_scan, axis=0)
    
    # Make prediction
    prediction = model.predict(processed_scan)
    
    return prediction

def predict_activations(scan_path, model = intermediate_layer_model):
    processed_scan = process_scan(scan_path)
    if processed_scan is None:
        return None, None  # Handle error if processing fails
    processed_scan = np.expand_dims(processed_scan, axis=0)
    
    activations = model.predict(processed_scan)
    
    return activations

# Example usage:
# Replace 'path_to_your_scan.nii' with the path to your actual CT scan
#scan_path = 'path_to_your_scan.nii'
#prediction = predict_single_scan(model, scan_path)

#if prediction is not None:
#    print("Prediction:", prediction)
#else:
#    print("Prediction could not be made. Please check the processing steps.")
