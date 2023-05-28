# Load libraries
library(keras)
library(neurobase)
library(tidyverse)

# Set the paths to your segmented images and labels folder
segmented_images_folder <- "./data/small_batch_test/"

# Load the segmented images
image_files <- list.files(segmented_images_folder, full.names = TRUE)
images <- lapply(image_files, readnii)

# Find the maximum z-dimension
max_z <- max(sapply(images, function(img) dim(img)[3]))

# Pad the images to the maximum z-dimension
images_padded <- lapply(images, function(img) {
  pad_width <- max_z - dim(img)[3]
  pad_dims <- c(0, 0, 0, pad_width)
  padded_img <- abind(img, array(0, dim = pad_dims))
  return(padded_img)
})

# Extract the images and labels from the loaded files
images_array <- array_reshape(images_padded, dim = c(dim(images_padded[[1]]), length(images_padded)))
labels_array <- array_reshape(labels, dim = c(dim(labels[[1]]), length(labels)))

# Normalize the images (adjust to your preferred normalization method)
images_array <- images_array / 255

# Split the data into training and testing sets
set.seed(42)
train_indices <- sample(length(images), floor(0.8 * length(images)))

train_images <- images_array[,,train_indices]
train_labels <- labels_array[,,train_indices]

test_images <- images_array[,, -train_indices]
test_labels <- labels_array[,, -train_indices]

# Define the model architecture
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = dim(train_images)[1:3]) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

# Train the model
history <- model %>% fit(
  train_images, train_labels,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# Evaluate the model on the test set
test_loss <- model %>% evaluate(test_images, test_labels)

# Print the test loss and accuracy
cat("Test loss:", test_loss$loss, "\n")
cat("Test accuracy:", test_loss$accuracy, "\n")