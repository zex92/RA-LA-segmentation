import pytorch_lightning as pl
import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import os


class LightningModel(pl.LightningModule):
    def __init__(self, model, model_dir):
        super().__init__()
        self.model = model
        self.loss = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
        #self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.total_val_loss = 0
        self.total_val_items = 0
        self.model_dir = model_dir

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def dice_metric(predicted, target):
        dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
        value = 1 - dice_value(predicted, target).item()
        return value

    def training_step(self, batch, batch_idx):
        volume, label = batch["image"], batch["label"]

        label = label != 0
        outputs = self(volume)
        train_loss = self.loss(outputs, label)
        train_metric = self.dice_metric(outputs, label)
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["image"].size(0))
        self.log('train_dice', train_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["image"].size(0))
        return train_loss

    def validation_step(self, batch, batch_idx):
        volume, label = batch["image"], batch["label"]
        label = label != 0
        outputs = self(volume)
        val_loss = self.loss(outputs, label)
        val_metric = self.dice_metric(outputs, label)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["image"].size(0))
        self.log('val_dice', val_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["image"].size(0))
        loss = self.loss(outputs, label)
        self.total_val_loss += loss.sum().item()
        self.total_val_items += len(outputs)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def on_validation_epoch_end(self):
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(self.total_val_loss / self.total_val_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
            torch.save(self.state_dict(), os.path.join(
                self.model_dir, "best_metric_model.pth"))
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        #self.log_dict(tensorboard_logs)
        self.total_val_loss = 0
        self.total_val_items = 0
