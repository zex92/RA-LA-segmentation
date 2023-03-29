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

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False).cuda()
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