import os
from glob import glob

from monai.transforms import (
    Compose,
    LoadImaged,  # Load images from a dictionary
    ToTensord,  # will be used to convert
    # will allow you to add new dimensions to your array
    EnsureChannelFirstd,
    # spaciing d
    Spacingd,  # allows us to change the spacing
    ScaleIntensityRanged,
    CropForegroundd,
    Resized, AddChanneld, Orientationd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandSpatialCropd,
    RandFlipd,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised

)

from monai.data import Dataset, DataLoader, CacheDataset
from monai.utils import first
import matplotlib.pyplot as plt


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
    # data_dir = 'C:/Users/Omar/Task02_Heart'
    # joins multiple paths


    if left == True:
        training_images = sorted(glob(os.path.join(data_dir, 'train_file_LA/images', '*.nii')))

        training_labels = sorted(
            glob(os.path.join(data_dir, 'train_file_LA/labels', '*.nii')))  # -> change to LA_labels was labels
        val_images = sorted(glob(os.path.join(data_dir, 'test_file_LA/images', '*.nii')))
        print(val_images)

        val_labels = sorted(glob(os.path.join(data_dir, 'test_file_LA/labels', '*.nii')))  # dont have
    else:
        training_images = sorted(glob(os.path.join(data_dir, 'train_file_RA/images', '*.nii')))
        training_labels = sorted(
            glob(os.path.join(data_dir, 'train_file_RA/labels', '*.nii')))  # -> chnage to RA_labels was labels
        val_images = sorted(glob(os.path.join(data_dir, 'test_file_RA/images', '*.nii')))

        val_labels = sorted(glob(os.path.join(data_dir, 'test_file_RA/labels', '*.nii')))  # dont have


    train_files = [{'image': image_name, 'label': label_name}
                   for image_name, label_name in zip(training_images, training_labels)]
    # for validation data
    val_files = [{'image': image_name, 'label': label_name}
                 for image_name, label_name in zip(val_images, training_labels)]

    # load images
    # do any transformations
    # need to convert them into torch sensors

    # compose allows you to perform multiple

    original_transformation = Compose(
        [
            LoadImaged(keys=['image', 'label']),
            # AddChanneld(keys=['image','label']),
            EnsureChannelFirstd(keys=['image', 'label']),  # adds a channel
            ToTensord(keys=['image', 'label'])

        ]
    )

    train_transforms = Compose(
        [
            LoadImaged(keys=['image', 'label']),  # needs to be first
            # AddChanneled(keys=['image','label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys="image"),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),  # try 5,5,5
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys='image', a_min=0, a_max=100, b_min=0.0, b_max=1.0, clip=True),
            # b needs to between 0 and 1 need to be float , because you needed it in order to deeplearn
            # how to know where which values to have for a you play with itk until you get a good-looking value
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False), #100,100,100
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), # remove (tweak changes)
            RandAdjustContrastd(keys=["image", "label"],prob=0.5,gamma=(0.5,4.5)), # change 1.5 ,2 at prob 1 (0.5 , 4.5)
            RandGaussianNoised(keys="image",prob=0.5, mean=0.0, std=0.1), # gave me 75.9 change back if worse
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), #divisor=10 , subtrahend=0
            # gets rid of dforeground from an image , gets rid of the image only
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]), #100 , 100 , 100
            ToTensord(keys=['image', 'label'])  # needs to be the last part


        ]
    )

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
    )


    # applies a transformer on the training files
    """
    original_ds = Dataset(data=train_files, transform=original_transformation)
    
    original_loader = DataLoader(original_ds, batch_size=1)"""
    if datacache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms)

        train_loader = DataLoader(train_ds, batch_size=1,shuffle=True)

        val_ds = CacheDataset(data=val_files, transform=val_transforms)

        val_loader = DataLoader(val_ds, batch_size=1,shuffle=True)
        return train_loader, val_loader

    else:

        train_ds = Dataset(data=train_files, transform=train_transforms)

        train_loader = DataLoader(train_ds, batch_size=1,shuffle=True)

        val_ds = Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1,shuffle=True)

        return train_loader, val_loader



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
            LoadImaged(keys=['image', 'label']),  # needs to be first
            # AddChanneled(keys=['image','label']),
            EnsureChannelFirstd(keys=['image', 'label']),
            EnsureTyped(keys="image"),
            #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2)),  # try 5,5,5
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys='image', a_min=0, a_max=100, b_min=0.0, b_max=1.0, clip=True),
            # b needs to between 0 and 1 need to be float , because you needed it in order to deeplearn
            # how to know where which values to have for a you play with itk until you get a good-looking value
            RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False), #100,100,100
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), # remove (tweak changes)
            RandAdjustContrastd(keys=["image", "label"],prob=0.5,gamma=(0.5,4.5)), # change 1.5 ,2 at prob 1 (0.5 , 4.5)
            RandGaussianNoised(keys="image",prob=0.5, mean=0.0, std=0.1), # gave me 75.9 change back if worse
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True), #divisor=10 , subtrahend=0
            # gets rid of dforeground from an image , gets rid of the image only
            Resized(keys=['image', 'label'], spatial_size=[128, 128, 64]), #100 , 100 , 100
            ToTensord(keys=['image', 'label'])  # needs to be the last part
"""