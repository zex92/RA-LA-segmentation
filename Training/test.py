from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    CropForegroundd,
    Activations, EnsureChannelFirstd,CastToTyped,NormalizeIntensityd
)
import nibabel as nib
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
from vnet import VNet
import torch
import matplotlib.pyplot as plt

import os
from glob import glob
import numpy as np
from PIL import Image


from monai.inferers import sliding_window_inference

value=input("Do you want to test the left of right atrium model?:")
left=bool()
if value.lower()=="left":
    left=True
else :
    right=True


in_dir="C:/Users/Admin/Heart_project/Heart_segmentation/DATA"

if left==True:
        model_dir = "C:/Users/Admin/Heart_project/left_atrium_data/results_LA"
else:
    model_dir = "C:/Users/Admin/Desktop/left_atrium_data/results_RA"  # change

train_loss = np.load(os.path.join(model_dir, 'loss_train.npy'))
train_metric = np.load(os.path.join(model_dir, 'metric_train.npy'))
test_loss = np.load(os.path.join(model_dir, 'loss_test.npy'))
test_metric = np.load(os.path.join(model_dir, 'metric_test.npy'))

plt.figure("Results of 9/02/2023", (12, 6))
plt.subplot(2, 2, 1)
plt.title("Train dice loss")
x = [i + 1 for i in range(len(train_loss))]
y = train_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 2)
plt.title("Train metric DICE")
x = [i + 1 for i in range(len(train_metric))]
y = train_metric
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 3)
plt.title("Test dice loss")
x = [i + 1 for i in range(len(test_loss))]
y = test_loss
plt.xlabel("epoch")
plt.plot(x, y)

plt.subplot(2, 2, 4)
plt.title("Test metric DICE")
x = [i + 1 for i in range(len(test_metric))]
y = test_metric
plt.xlabel("epoch")
plt.plot(x, y)
plt.tight_layout()
plt.show()




# need to change

if left==True:
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "train_file_LA/labels", "*.nii"))) # .gz for zipped  #
    path_train_volumes = sorted(glob(os.path.join(in_dir, "train_file_LA/images", "*.nii")))
    path_test_volumes = sorted(glob(os.path.join(in_dir, "test_file_LA/images", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "test_file_LA/labels", "*.nii")))
else :
    path_train_volumes = sorted(glob(os.path.join(in_dir, "train_file_RA/images", "*.nii")))
    path_train_segmentation = sorted(glob(os.path.join(in_dir, "train_file_RA/labels", "*.nii"))) # .gz for zipped  #
    path_test_volumes = sorted(glob(os.path.join(in_dir, "test_file_RA/images", "*.nii")))
    path_test_segmentation = sorted(glob(os.path.join(in_dir, "test_file_RA/labels", "*.nii")))



train_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_train_volumes, path_train_segmentation)]
test_files = [{"vol": image_name, "seg": label_name} for image_name, label_name in zip(path_test_volumes, path_test_segmentation)]
test_files = test_files[0:9]
print(test_files)


keys = ["vol", "seg"]
test_transforms = Compose(
    [
        LoadImaged(keys),
        EnsureChannelFirstd(keys=['vol', 'seg']),
        Orientationd(keys, axcodes="RAS"),
        Spacingd(keys, pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),
        CastToTyped(keys, dtype=np.float32),
        #CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        #NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
        Resized(keys=['vol', 'seg'], spatial_size=[128, 128, 64]),
        ToTensord(keys)
    ]
)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128,258),
    strides=(2, 2, 2,2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)



#model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth")))


"""model.load_state_dict(torch.load(
    "C:/Users/Admin/Heart_project/Heart_segmentation/iter_6000.pth"))"""
model.eval()


sw_batch_size = 4
roi_size = (128, 128, 64)
with torch.no_grad():
    test_patient = first(test_loader)

    t_volume = test_patient['vol']
    #t_volume = test_patient['seg']
    test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
    sigmoid_activation = Activations(sigmoid=True)
    test_outputs = sigmoid_activation(test_outputs)
    test_outputs = test_outputs > 0.53




"""
array = np.array(test_outputs.detach().cpu()).astype(int)


for i, c in enumerate(range(0, 550, 10)):
    array[0, 1, :, :, i] *= c

input_image = nib.load("C:/Users/Admin/Heart_project/Heart_segmentation/DATA/test_file_LA/images/la_0101.nii")
original_shape = input_image.shape




header = nib.Nifti1Header(endianness='<')
header.set_data_shape((640, 640, 44))
header.set_data_dtype(np.float32)

header['sizeof_hdr'] = 348
header['data_type'] = b''
header['db_name'] = b''
header['extents'] = 0
header['session_error'] = 0
header['regular'] = b'r'
header['dim_info'] = 0
header['intent_p1'] = 0.0
header['intent_p2'] = 0.0
header['intent_p3'] = 0.0
header['intent_code'] = 0
header['bitpix'] = 32
header['slice_start'] = 0
header['pixdim'] = [1.0, 0.625, 0.625, 2.5, 0.0, 0.0, 0.0, 0.0]
header['vox_offset'] = 0.0
header['scl_slope'] = np.nan
header['scl_inter'] = np.nan
header['slice_end'] = 0
header['slice_code'] = 0
header['xyzt_units'] = 2
header['cal_max'] = 0.0
header['cal_min'] = 0.0
header['slice_duration'] = 0.0
header['toffset'] = 0.0
header['glmax'] = 0
header['glmin'] = 0
header['descrip'] = b''
header['aux_file'] = b''
header['qform_code'] = 0
header['sform_code'] = 2
header['quatern_b'] = 0.0
header['quatern_c'] = 0.0
header['quatern_d'] = 0.0
header['qoffset_x'] = -200.0
header['qoffset_y'] = -170.944
header['qoffset_z'] = -17.9146
header['srow_x'] = [0.625, 0.0, 0.0, -200.0]
header['srow_y'] = [0.0, 0.625, 0.0, -170.944]
header['srow_z'] = [0.0, 0.0, 2.5, -17.9146]
header['intent_name'] = b''
header['magic'] = b'n+1'

affine = np.array([[ 0.625, 0.0, 0.0, -200.0],
                   [0.0, 0.625, 0.0, -170.944],
                   [0.0, 0.0, 2.5, -17.9146],
                   [0.0, 0.0, 0.0, 1.0]])

nifti_image = nib.Nifti1Image(array,affine)
for key, value in header.items():
    nifti_image.header[key] = value

print(nifti_image)
nib.save(nifti_image, "C:/Users/Admin/Heart_project/Heart_segmentation/DATA/array_nifti/fileleeee.nii")

la_101=nib.load("C:/Users/Admin/Heart_project/Heart_segmentation/DATA/test_file_LA/images/la_0101.nii")
"""


for i in range(0,60):
        # plot the slice [:, :, 80]
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title(f"image {i}")
        plt.imshow(test_patient["vol"][0, 0, :, :, i], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f"label {i}")
        plt.imshow(test_patient["seg"][0, 0, :, :, i] != 0)
        plt.subplot(1, 3, 3)
        plt.title(f"output {i}")
        plt.imshow(test_outputs.detach().cpu()[0, 1, :, :, i])
        plt.show()