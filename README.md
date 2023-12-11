# Predicting AF outcomes using 3D deep learning of CT scans

## Dataset Building
### DICOM to NifTI
The input scans are in the DICOM format, and are converted to NifTI images using `dicom2nii.py`. A single DICOM file can contain multiple image slices or volumes within it, each representing different aspects or sequences of medical imaging data. When converting a DICOM file to NIfTI, each of these slices or volumes is extracted and saved as separate NIfTI images. Console output is stored in `data/log/dicom2nii.out`.

### Best NifTI Selection
The best NifTI image is defined as the image with the best spacing. `bestnifti.py` selects this image based on a lookup table of previously generated images (`data/vandyniftibest.csv`). Console output is stored in `data/log/bestnifti.out`.

### Preprocessing
`preprocess.py` reads in the best NifTI images, normalizes it, and resizes them to a consistent size. The processed NifTI images are test-train split based on a clinical outcome (defined in `data/vanderbilt_ct_phenotype_2-14-23.csv`) and are saved to `data.pkl`. 

## Modeling
`model.py` trains a model from `data.pkl` and saves the best model (measured by validation accuracy) to `best_classifier.h5`. The model architecture is defined here:
Model: "3dcnn"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_4 (InputLayer)        [(None, 128, 128, 64, 1   0         
                             )]                                  
                                                                 
 conv3d_8 (Conv3D)           (None, 126, 126, 62, 64   1792      
                             )                                   
                                                                 
 max_pooling3d_8 (MaxPoolin  (None, 63, 63, 31, 64)    0         
 g3D)                                                            
                                                                 
 batch_normalization_8 (Bat  (None, 63, 63, 31, 64)    256       
 chNormalization)                                                
                                                                 
 conv3d_9 (Conv3D)           (None, 61, 61, 29, 64)    110656    
                                                                 
 max_pooling3d_9 (MaxPoolin  (None, 30, 30, 14, 64)    0         
 g3D)                                                            
                                                                 
 batch_normalization_9 (Bat  (None, 30, 30, 14, 64)    256       
 chNormalization)                                                
                                                                 
 conv3d_10 (Conv3D)          (None, 28, 28, 12, 128)   221312    
                                                                 
 max_pooling3d_10 (MaxPooli  (None, 14, 14, 6, 128)    0         
 ng3D)                                                           
                                                                 
 batch_normalization_10 (Ba  (None, 14, 14, 6, 128)    512       
 tchNormalization)                                               
                                                                 
 conv3d_11 (Conv3D)          (None, 12, 12, 4, 256)    884992    
                                                                 
 max_pooling3d_11 (MaxPooli  (None, 6, 6, 2, 256)      0         
 ng3D)                                                           
                                                                 
 batch_normalization_11 (Ba  (None, 6, 6, 2, 256)      1024      
 tchNormalization)                                               
                                                                 
 global_average_pooling3d_2  (None, 256)               0         
  (GlobalAveragePooling3D)                                       
                                                                 
 dense_4 (Dense)             (None, 512)               131584    
                                                                 
 dropout_2 (Dropout)         (None, 512)               0         
                                                                 
 dense_5 (Dense)             (None, 1)                 513       
                                                                 
=================================================================
Total params: 1352897 (5.16 MB)
Trainable params: 1351873 (5.16 MB)
Non-trainable params: 1024 (4.00 KB)
_________________________________________________________________