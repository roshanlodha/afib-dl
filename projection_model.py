import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense Lambda

af_recur = np.fromfile('./cache/af_recur.dat', dtype=int)
preprocessed_images = np.fromfile('./cache/projected_data.dat', dtype=int)

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

model.save('./cache/model')