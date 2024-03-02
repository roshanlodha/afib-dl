import os
import SimpleITK as sitk
import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

source_directory = '/proj/genetics/Projects/shared/Subject Sources/CCF/Cardiac_Images/Data/Images/CT_Images/Pulled_Before_2022/'
destination_directory = '/home/lodhar/afib-dl/nifti/ccfniftiall/'

if not os.path.exists(destination_directory):
    os.mkdir(destination_directory)
"""
for root, sub, files in os.walk(source_directory):
    if not sub and files:
        if len(files)>60:
            print(root)
            code=root.split('Pulled_Before_2022/')[1].split('/')[0]
            print(code)
            new_path = destination_directory + code
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            try:
                dicom2nifti.convert_directory(root, new_path)
            except Exception as e:
                pass
"""
for root, sub, files in os.walk(source_directory):
    if not sub and files:
        if len(files)>60:
            code = root.split('Pulled_Before_2022/')[1].split('/')[0]
            print(code)
            
            new_path = destination_directory + code
            if not os.path.exists(new_path):
                os.mkdir(new_path)

            new_file = new_path + "/" + code + '.nii.gz'

            try:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                sitk.WriteImage(image, new_file)
            except:
                pass