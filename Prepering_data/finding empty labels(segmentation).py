import nibabel as nib
import numpy as np
from glob import glob


# will be used to check if a label is empty
# checks if a label is 0 means its empty

# loads the label data
input_nifti_file_path="C:/Users/Omar/Desktop/left_atrium_data/LA_labels"
# path contains multiple files

list_labels=glob(input_nifti_file_path)

# will loop for evey patient
for patient in list_labels:

    nifty_file=nib.load(patient)
    # this loads the images

    # will contain an array of the slices  will be used to check the pixels
    # if contains all 0 that means there is no heart
    fdata=nifty_file.get_fdata()

    np_unique=np.unique(fdata)

    # checks if its empty
    if len(np_unique) == 1 :
        print(patient)
        # can delete file bu using shantil.delete pass patient




