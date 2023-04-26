
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    CropForegroundd,
    EnsureChannelFirstd, CastToTyped, NormalizeIntensityd,Activations
)
import nibabel as nib
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import DataLoader, Dataset
from vnet import VNet
import torch
from monai.inferers import sliding_window_inference
import os
from glob import glob
import numpy as np
from monai.data import MetaTensor


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
    model_dir = "C:/Users/Admin/Heart_project/left_atrium_data/results_RA"


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

keys = ["vol", "seg"]
test_transforms = Compose(
    [
        LoadImaged(keys),
        EnsureChannelFirstd(keys=['vol', 'seg']),
        Orientationd(keys, axcodes="RAS"),#
        Spacingd(keys, pixdim=(1.5, 1.5, 2), mode=("bilinear", "nearest")),#
        CastToTyped(keys, dtype=np.float32),
        CropForegroundd(keys=['vol', 'seg'], source_key='vol'),
        NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
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


model= VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth")))



model.eval()


sw_batch_size = 4
roi_size = (128, 128, 64)
with torch.no_grad():
    test_patient = first(test_loader)
    t_volume = test_patient['vol']
    test_outputs = sliding_window_inference(t_volume.to(device), roi_size, sw_batch_size, model)
    sigmoid_activation = Activations(sigmoid=True)
    test_outputs = sigmoid_activation(test_outputs)
    test_outputs = test_outputs.detach().cpu()
    test_outputs = torch.tensor(np.logical_not(test_outputs.numpy() > 0.6), dtype=torch.float32)
    test_outputs = MetaTensor(test_outputs)




test_label=np.array(test_patient["seg"] !=0).astype(int)
array = np.array(test_outputs.detach().cpu()).astype(int)


test_label= torch.from_numpy(test_label)
test_label=test_label.permute(2,3,4,0,1)
print(f"{test_label.shape} this is the test_labels")
array= torch.from_numpy(array)
array=array.permute(2,3,4,0,1)

print(array.shape)

test_label=np.array(test_label)
array=np.array(array)

nifti_image = nib.Nifti1Image(array,np.eye(4))
test_nifti=nib.Nifti1Image(test_label,np.eye(4))


nib.save(test_nifti, "C:/Users/Admin/Heart_project/Heart_segmentation/DATA/array_nifti/test_label")
nib.save(nifti_image, "C:/Users/Admin/Heart_project/Heart_segmentation/DATA/array_nifti/predicted_label")

