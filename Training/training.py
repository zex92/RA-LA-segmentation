from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss,DiceCELoss
from vnet import VNet

import os

import torch

from PreProcessing.preparingData import prepering
from Training.utilities import train

from LightningModel import LightningModel


# 2. Set `python` built-in pseudo-random generator at a fixed value
"""import random

seed_value=123
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# 4. Set `tensorflow` pseudo-random generator at a fixed value
import torch
torch.cuda.manual_seed(seed_value)
#torch.seed(seed_value)
"""
left = bool()
value = input("Do you want to train the left or right atrium?:")
if value.lower() == "left":
    left = True
else:
    left = False

# data_dir = "C:/Users/Omar/Task02_Heart" # change to new directory  -> C:/Users/Omar/Desktop/left_atrium_data


# model_dir = "C:/Users/Omar/Task02_Heart/results" # change  -> C:/Users/Omar/Desktop/left_atrium_data/results
data_dir="C:/Users/Admin/Heart_project/Heart_segmentation/DATA"

if left == True:
    model_dir = "C:/Users/Admin/Heart_project/left_atrium_data/results_LA"
else:
    model_dir = "C:/Users/Admin/Heart_project/left_atrium_data/results_RA"

data_in = prepering(data_dir, left=left,datacache=True)  # prepares (transform the data) so It's ready to be trained

device = "cuda" if torch.cuda.is_available() else "cpu"  # used to allow me to train data on GPU when i have one

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128,258),
    strides=(2, 2, 2,2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
"""vnet_model = VNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    act=("elu", {"inplace": True}),
    dropout_prob=0.5,
    dropout_dim=3,
    bias=True
).to(device)"""

"""model = LightningModel(vnet_model,model_dir)
model.to(device)"""

"""model = VNet(  # create the model to be used to train the data
    spatial_dims=3,in_channels=1, out_channels=2
).to(device)"""


"""
model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True).cuda()
"""

model.load_state_dict(torch.load(
    os.path.join(model_dir, "best_metric_model.pth")))
    

#model.load_state_dict(torch.load("C:/Users/Admin/Heart_project/Heart_segmentation/iter_6000.pth"))
"""




"""
#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))"""
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
# the loss function thats used to test how my model is doing

#optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

#optimizer = torch.optim.Adam(model.parameters(), 0.0001, weight_decay=5e-5, amsgrad=True)
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
# this is used to apply back propagation and improve my model

if __name__ == '__main__':
    #k_fold_cross_validation(data_in,model,loss_function,optimizer)
    train(model, data_in, loss_function, optimizer,200, model_dir)
    #train(model, data_in, 1, model_dir)
    #print(f"train completed, best_metric: {model.best_val_dice:.4f} " f"at epoch {model.best_val_epoch}")