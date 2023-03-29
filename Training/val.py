from monai.utils import first, set_determinism
from monai.transforms import (
    Compose,
    AddChanneld,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    Activations, EnsureChannelFirstd, EnsureTyped, NormalizeIntensityd,
)

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset
from vnet import VNet
import torch
import matplotlib.pyplot as plt

import os
from glob import glob

from tqdm import  tqdm
import numpy as np
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


model_dir = "C:/Users/Admin/Heart_project/left_atrium_data/results_LA"

model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth")))
device = "cuda" if torch.cuda.is_available() else "cpu"  # used to allow me to train data on GPU when i have one
test_transforms = Compose(
    [
        LoadImaged(keys=['vol', 'seg']),  # needs to be first
        EnsureChannelFirstd(keys=['vol', 'seg']),
        EnsureTyped(keys="vol"),
        #ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["vol", "seg"], axcodes="RAS"),
        Spacingd(keys=['vol', 'seg'], pixdim=(1.5, 1.5, 2)),
        NormalizeIntensityd(keys="vol", nonzero=True, channel_wise=True),
        Resized(keys=['vol', 'seg'], spatial_size=[128, 128, 64]),
        ToTensord(keys=['vol', 'seg'])  # needs to be the last part
    ]
)

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1)


y_preds=[]
model.eval()

sw_batch_size = 4
roi_size = (128, 128, 64)
with torch.inference_mode():
    test_patient = first(test_loader)
    for X,y in tqdm(zip(test_patient["vol"], test_patient["seg"]),desc="Making predictions ..."):
        test_outputs = sliding_window_inference(X.to(device), roi_size, sw_batch_size, model)
        sigmoid_activation = Activations(sigmoid=True)
        test_outputs = sigmoid_activation(test_outputs)
        y_preds.append(test_outputs.cpu())

y_pred_tensor=torch.cat(y_preds)

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

print(y_pred_tensor.shape)
print(np.array(test_ds).shape)
print(test_patient["vol"][0, 0, :, :, 0].shape)
confmat=ConfusionMatrix(num_classes=2,task="multiclass")
confmat_tensor=confmat(preds=y_pred_tensor,target=np.array(zip(test_patient["vol"], test_patient["seg"])))
# create and make into a tensor
fig,ax=plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names=2,
    figsize=(10,7)
)
