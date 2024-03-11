import os
import dicom2nifti
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()

for root, sub, files in os.walk('/proj/genetics/Projects/shared/Subject Sources/External/Vanderbilt_CT_fromCWRU/Data/'):
    if not sub and files:
        if len(files)>60:
            print(root)
            code=root.split('AFR.')[1].split('/')[0]
            print(code)
            subcode=root[-5::]
            print(subcode)
            new_path='/home/lodhar/vandynifti/'+code
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            new_path='/home/lodhar/vandynifti/'+code+'/'+subcode
            if not os.path.exists(new_path):
                os.mkdir(new_path)
            dicom2nifti.convert_directory(root, new_path)