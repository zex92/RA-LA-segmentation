from glob import glob
import shutil
import os
import gzip


"""
1.get the data file (main start) done 
2.move unzip the data 
"""

file="C:/Users/Admin/Heart_project/Heart_segmentation/dataset"
task2=glob(file+"/task2")
train_task_2=sorted(glob(file+"/task2/train_data/*")) # has all the folders


for file in train_task_2:
    print(file)
    print("\n")

def unzip_gz():
    for i,data in  enumerate(train_task_2):

        data_path=os.path.normpath(data)

        #print(os.path.normpath(data))
        image_path=str(data_path)+"/atriumSegImgMO.nii.gz"
        label_path=data_path+"/enhanced.nii.gz"
        #if i==0:
        #print(str("C:\Users\Omar\IdeaProjects\Heart_Segmentation\Heart_segmentation\dataset\task2\train_data\train_1/atriumSegImgMO.nii.gz") == image_path)
        print(image_path)

        #gzip.compress(data_path)
        with gzip.open(image_path, 'rb') as f_in:
            with open(data_path+"/atriumSegImgMO.nii", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                #os.remove(image_path)


        with gzip.open(label_path,"rb") as f:
            with open(data_path+"/enhanced.nii","wb") as f_o:
                shutil.copyfileobj(f,f_o)
                #os.remove(label_path)

        #print(f"{data} is number {i}\n")



for i,data in  enumerate(train_task_2):

    data_path=os.path.normpath(data)
    print(data_path)
    image_path=str(data_path)+"/atriumSegImgMO.nii"
    label_path=data_path+"/enhanced.nii"





for i, data in enumerate(train_task_2):

    i+=33

    image_out_train="C:/Users/Admin/Heart_project/Heart_segmentation/DATA/train_file_LA/images"
    label_out_train="C:/Users/Admin/Heart_project/Heart_segmentation/DATA/train_file_LA/labels"
    image_out_test="C:/Users/Admin/Heart_project/Heart_segmentation/DATA/test_file_LA/images"
    label_out_test="C:/Users/Admin/Heart_project/Heart_segmentation/DATA/test_file_LA/labels"

    data_path=os.path.normpath(data)
    print(data_path)
    image_path=str(data_path)+"/atriumSegImgMO.nii"
    label_path=data_path+"/enhanced.nii"

    if i<134:
        image_out_train = os.path.join(image_out_train,"la_0"+str(i)+".nii")
        label_out_train = os.path.join(label_out_train,"la_0"+str(i-2)+".nii")
        os.rename(image_path,image_out_train)
        os.rename(label_path,label_out_train)


    if i> 134:

        image_out_test = os.path.join(image_out_test,"la_0"+str(i-129)+".nii")
        label_out_test = os.path.join(label_out_test,"la_0"+str(i-129)+".nii")
        os.rename(image_path,image_out_test)
        os.rename(label_path,label_out_test)

        #print(image_out_test)




