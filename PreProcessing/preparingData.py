import os
from glob import glob

from monai.transforms import (
    # will be used to convert
    # will allow you to add new dimensions to your array
    EnsureChannelFirstd,
    Resized, AddChanneld, Orientationd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Rand3DElasticd,
    Compose,
    LoadImaged,
    Spacingd,
    RandCropByPosNegLabeld,
    Orientationd,
    CastToTyped,
    ToTensord,
    EnsureTyped,
    RandShiftIntensityd,
    NormalizeIntensityd,
    CropForegroundd
)

from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import first
import matplotlib.pyplot as plt
import numpy as np

def prepering(data_dir, datacache=False, left=True):
    """
    Purpose :

    Args:
    :param data_dir:
        the data_dir is a sting datatype , which contains the path to the where the images of the heart
        and the labels of both the right and left atrium is
    :param datacache:
        the date cache argument is a boolean argument which will be used to give the option to the user whether
        they would like to run on cache , which may speed up computation
    :param left:
        the left argument is a boolean value which will be used to check which labels will be used to train the model
        either the left atrium data or the right atrium data
    :return:
        it returns  a data loader which contains the transformed data (augmented and turned into tensors) this is
        important in order to be able to train the model.
    """
    # main part
    # data_dir = 'C:/Users/Admin/Task02_Heart'
    # joins multiple paths


    if left == True:
        training_images = sorted(glob(os.path.join(data_dir, 'train_file_LA/images', '*.nii')))

        training_labels = sorted(
            glob(os.path.join(data_dir, 'train_file_LA/labels', '*.nii')))  # -> change to LA_labels was labels
        test_images = sorted(glob(os.path.join(data_dir, 'test_file_LA/images', '*.nii')))
        test_labels = sorted(glob(os.path.join(data_dir, 'test_file_LA/labels', '*.nii')))  # dont have

        val_images=sorted(glob(os.path.join(data_dir, 'val_file_LA/images', '*.nii')))
        val_labels=sorted(
            glob(os.path.join(data_dir, 'val_file_LA/labels', '*.nii')))


    else:
        training_images = sorted(glob(os.path.join(data_dir, 'train_file_RA/images', '*.nii')))
        training_labels = sorted(
            glob(os.path.join(data_dir, 'train_file_RA/labels', '*.nii')))  # -> chnage to RA_labels was labels
        test_images = sorted(glob(os.path.join(data_dir, 'test_file_RA/images', '*.nii')))

        test_labels = sorted(glob(os.path.join(data_dir, 'test_file_RA/labels', '*.nii')))  # dont have
        val_images=sorted(glob(os.path.join(data_dir, 'val_file_RA/images', '*.nii')))
        val_labels=sorted(
            glob(os.path.join(data_dir, 'val_file_LA/labels', '*.nii')))








    train_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(training_images, training_labels)]
    # for validation data
    test_files = [{'image': image_name, 'label': label_name}
                 for image_name, label_name in zip(test_images, test_labels)]

    val_files= [{'image': image_name, 'label': label_name}
                for image_name, label_name in zip(val_images, val_labels)]


    keys = ["image", "label"]
    train_transforms = Compose(
        [
            LoadImaged(keys),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys, axcodes="RAS"),
            Spacingd(keys, pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),
            CastToTyped(keys, dtype=np.float32),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandAffined(keys, prob=0.1, rotate_range=30, translate_range=5, scale_range=(0.9, 1.1)),
            RandFlipd(keys, prob=0.5, spatial_axis=[0, 1, 2]),
            RandGaussianNoised("image", prob=0.5, mean=0.0, std=0.05),
            Rand3DElasticd("image", prob=0.1, sigma_range=(1, 3), magnitude_range=(0, 0.5)),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys)
        ]
    )
    test_transforms = Compose(
        [
            LoadImaged(keys),
            EnsureChannelFirstd(keys=['image', 'label']),
            Orientationd(keys, axcodes="RAS"),
            Spacingd(keys, pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),
            CastToTyped(keys, dtype=np.float32),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys)
        ]
    )

    val_transforms= Compose(
        [
            LoadImaged(keys),
            EnsureChannelFirstd(keys=['image', 'label']),
            Spacingd(keys,pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),  # Resample to 1mm x 1mm x 1mm voxel size
            RandCropByPosNegLabeld(keys, label_key="label", spatial_size=[128, 128, 64], num_samples=1, pos=1, neg=1),
            Orientationd(keys, axcodes="RAS"),
            CastToTyped(keys, dtype=np.float32),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys)
        ]
    )


    # applies a transformer on the training files
    """
    original_ds = Dataset(data=train_files, transform=original_transformation)
    
    original_loader = DataLoader(original_ds, batch_size=1)"""
    if datacache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms,num_workers=16)

        train_loader = DataLoader(train_ds)

        test_ds = CacheDataset(data=test_files, transform=test_transforms,num_workers=16)

        test_loader = DataLoader(test_ds)

        #val_ds = CacheDataset(data=val_files, transform=val_transforms)

        #val_loader = DataLoader(val_ds, batch_size=1,shuffle=True)

        return train_loader, test_loader

    else:
        train_ds = Dataset(data=train_files, transform=train_transforms,num_workers=16)

        train_loader = DataLoader(train_ds)

        test_ds = Dataset(data=test_files, transform=test_transforms,num_workers=16)
        test_loader = DataLoader(test_ds)

        #val_ds = Dataset(data=val_files, transform=val_transforms)

        #val_loader = DataLoader(val_ds, batch_size=1,shuffle=True)


        return train_loader, test_loader



def show_patient(data, SLICE_NUMBER=50, train=True, test=False):
    """

    :param data:
        this takes in a dataloader , which will contain train and test data
    :param SLICE_NUMBER:
        The SLICE_NUMBER is an int which dictates which slice to plot
    :param train:
        the train is a boolean datatype which dictates whether to show the train dataloader
    :param test:
        the train is a boolean datatype which dictates whether to show the test dataloader
    :return:
        doesn't return anything :
        but its used to plot the image of the heart image (this function was used to test whether my data
        transformation was done correctly)
    """
    check_patient_train, check_patient_test = data

    train_patient = first(check_patient_test)

    test_patient = first(check_patient_test)
    if train:
        plt.figure('Train', (12, 6))

        plt.subplot(1, 2, 1)  # divides into 3
        plt.title('Slice of a patient')

        # b= batch size , w=width , h= height , s=slice  , c= channel / background
        plt.imshow(train_patient['image'][0, 0, :, :, SLICE_NUMBER], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Label of a patient')

        plt.imshow(train_patient['label'][0, 0, :, :, SLICE_NUMBER])

        plt.show()

    if test:  # need to change into
        plt.figure('Test', (12, 6))

        plt.subplot(1, 2, 1)  # divides into 3
        plt.title('Slice of a patient')

        # b= batch size , w=width , h= height , s=slice  , c= channel / background
        plt.imshow(train_patient['image'][0, 0, :, :, SLICE_NUMBER], cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Label of a patient')

        plt.imshow(train_patient['label'][0, 0, :, :, SLICE_NUMBER])

        plt.show()


"""
    val_transforms = Compose(
        [

            LoadImaged(keys=['image', 'label']),  # needs to be first
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys="image"),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys=['image', 'label'])  # needs to be the last par
        ]
    )"""


"""data=data_dir="C:/Users/Admin/Heart_project/Heart_segmentation/DATA"

train=prepering(data,left=False)

show_patient(train,test=True,train=False)


"""


"""
            LoadImaged(keys=["vol", "seg"]),
            AddChanneld(keys=["vol", "seg"]),
            Spacingd(keys=["vol", "seg"], pixdim=pixdim, mode=("bilinear", "nearest")),
            Orientationd(keys=["vol", "seg"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True), 
            CropForegroundd(keys=["vol", "seg"], source_key="vol"),
            Resized(keys=["vol", "seg"], spatial_size=spatial_size),   
            ToTensord(keys=["vol", "seg"]),
"""


"""
            LoadImaged(keys=['image', 'label']),  # needs to be first
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys="image"),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]),
            ToTensord(keys=['image', 'label'])  # needs to be the last par
"""


"""
        [
            EnsureTyped(keys="image"),
            #RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False), #100,100,100  made it torse
            #RandAdjustContrastd(keys=["image", "label"],prob=0.5,gamma=(0.5,4.5)), # change 1.5 ,2 at prob 1 (0.5 , 4.5)
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), #divisor=10 , subtrahend=0
            # gets rid of dforeground from an image , gets rid of the image only
        ]
"""
