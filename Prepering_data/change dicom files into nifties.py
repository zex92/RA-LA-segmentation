from Prepering_data import Processing
from glob import glob
#STEP2

import dicom2nifti
import os
# will be used to chnage dicom files into nifties

# dicom series  2 paramters

in_path_images = 'C:/Users/Omar/Task02_Heart/decomGroups/images/*'
# contains images and labels
in_path_labels= 'C:/Users/Omar/Task02_Heart/decomGroups/labels/*'

out_path_images='C:/Users/Omar/Task02_Heart/nifti_files/images'
out_path_labels='C:/Users/Omar/Task02_Heart/nifti_files/labels'



list_of_images= glob(in_path_images)
list_of_labels=glob(in_path_labels)

print(list_of_labels)

print(list_of_images)



# input , output= name of wheenre you want to save your file
#start with images and then labels
for patient in list_of_labels:
    # you need to save the index
    # Preprossesin.patiant_name =name of the patiat
    # add '.nii' = nifti files  if you want compresset  do .nii.gz
    patient_name=os.path.basename(os.path.normpath(patient))
    dicom2nifti.dicom_series_to_nifti(patient,os.path.join(out_path_labels,patient_name+'.nii.gz'))

    # to change the images (change the for loop and the out_path)


