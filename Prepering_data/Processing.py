from glob import glob
import shutil
import os



in_file = 'C:/Users/Omar/Task02_Heart/decomFiles/images'
out_file = 'C:/Users/Omar/Task02_Heart/decomGroups/images'


# returns the patient

for patient in glob(in_file + '/*'):# forgot the /
    # its * because we are passing a folder meaning our extention can be anything
    # glob returns a list of all the files in a directory / of the parts

    # return partisan name
    # basename is the last part of the file
    # os.path.normpath = normalises the path
    patient_name = os.path.basename(os.path.normpath(patient))
    print(patient_name)  # returns the baseline of the file

    # return  number of slices of a certain patient (return a list of all the slices)
    number_folders = int(len(glob(patient + '/*'))/10)
    # cretes folder to save the slices
    for i in range(number_folders):

        # create a new sub directory this will join paths with a certain name
        # output is a bit diffrent
        output_path_name = os.path.join(out_file, patient_name + '_' + str(i))
        print(output_path_name)

        os.mkdir(output_path_name) # makes directory for the path

        # how to move the slices  into the new folder
        for i, file in enumerate(glob(patient + '/*')):

            # break if reached number of slices
            if i == 10:
                break

            # create a (file for each file with a part of 110 slices)
            shutil.move(file , output_path_name) # moves file to directory we want



#chnage to images afrter you do the labels
