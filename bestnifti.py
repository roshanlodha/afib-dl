import sys
import os
import shutil

import pandas as pd

bestnifti = pd.read_csv('./vandyniftibest.csv').set_index('name')

for case, dicom in bestnifti['best'].items():
    bestaslist = list(dicom.replace("]", "").split(","))
    subdir = bestaslist[3].replace("'", "").replace(" ", "") + '/'
    filename = bestaslist[4].replace("'", "").replace(" ", "")
    bestfilepath = 'vandyniftiall/' + str(case) + '/' + subdir + filename
    destination = 'vandyniftibest/' + str(case) + '.nii.gz'
    if os.path.exists(destination):
        print(destination + " already exists")
    else:
        try:
            shutil.copy(bestfilepath, destination)
            print(destination + " copied successfully")
        except:
            print(destination + " not copied")