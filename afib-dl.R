library(neurobase)
library(fslr)

# Set the folder path where segmented images are stored
folder_path <- "data/small_batch_test/"

# Get the list of files in the folder with the .nii.gz extension
file_list <- list.files(folder_path, pattern = "\\.nii\\.gz$", full.names = TRUE)

# Read in and resample the segmented images
resampled_images <- lapply(file_list, function(file) {
  # Read the image using the neurobase function
  img <- readnii(file)
  
  # Resample the image using the fsl_resample function
  resampled_img <- fsl_resample(img, voxel_size = c(2, 2, 2))
  
  return(resampled_img)
})

# Find the maximum dimensions among the resampled images
max_dims <- max(sapply(resampled_images, function(img) dim(img)[1:3]))

# Pad the images to the same size
padded_images <- lapply(resampled_images, function(img) {
  padded_img <- neurobase::pad(img, targetdim = c(max_dims, max_dims, max_dims))
  return(padded_img)
})

# Access the resampled and padded images
# padded_images[[1]] for the first image, padded_images[[2]] for the second, and so on.
