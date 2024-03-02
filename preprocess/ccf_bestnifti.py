import sys
import os
import shutil

if not os.path.exists("/home/lodhar/afib-dl/nifti/ccf/"):
    os.makedirs("/home/lodhar/afib-dl/nifti/ccf/")

for root, sub, files in os.walk("/home/lodhar/afib-dl/nifti/ccfniftiall/"):
	if len(files) > 0:
		code = root.split("ccfniftiall/")[1]
		
		source = max((os.path.join(root, file) for root, dirs, files in os.walk(root) for file in files), key=os.path.getsize)
		#source = "/home/lodhar/afib-dl/nifti/ccfniftiall/" + code + "/" + code + ".nii.gz"
		
		dest = "/home/lodhar/afib-dl/nifti/ccf/" + code + ".nii.gz"
		
		try:
			shutil.copy(source, dest)
			print(code + " copied successfully.")
		except:
			print(code + " not copied.")
