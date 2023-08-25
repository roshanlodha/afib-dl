import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv3D, Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model

np.fromfile('./cache/resampled_data.dat', dtype=int)
np.fromfile('./cache/af_recur.dat', dtype=int)

# Load pre-trained VGG16 model without the top (fully connected) layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(None, None, None, 3))

# Freeze the weights of the pre-trained layers
for layer in base_model.layers:
    layer.trainable = False

# Add a 3D convolutional layer to the base model
x = base_model.output
x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(x)

# Add additional layers for your specific task
# Example: Flatten, fully connected layers, and output layer
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Modify the output layer according to your labels

# Create a new model
model = Model(inputs=base_model.input, outputs=output)

# Reshape the input data to add a grayscale channel and convert to RGB
resampled_data = np.reshape(resampled_data, (*resampled_data.shape, 1))
resampled_data = np.concatenate((resampled_data, resampled_data, resampled_data), axis=-1)

# Perform train-test-validation split
X_train, X_test, y_train, y_test = train_test_split(resampled_data, af_recur, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)  # Modify the training parameters as needed

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")