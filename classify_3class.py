import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, random_split
import os
# import pytorch_lightning as pl
import lightning.pytorch as pl
from pprint import pprint
import pdb
from lightning.pytorch.loggers import TensorBoardLogger
from clearml import Task
import torch 
from lightning.pytorch.callbacks import ModelCheckpoint

from data.BRTWDataset import BRTWDataset,Dataset_3Building
from model.unet import BRTWModel, PlModel, UNet
from callbacks.pl_visualize import VisualizeCallBack,VisualizeCallBack_fit

task = Task.init(
    project_name="3_bulding_bien",
    task_name="BRTWModel_Unet++_resnet34 Training",
    # output_uri=True  # IMPORTANT: setting this to True will upload the model
    # If not set the local path of the model will be saved instead!
)


ROOT_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
IMG_SIZE = (512,512)
BATCH_SIZE = 4
CLASSES = 5

checkpoint_callback = ModelCheckpoint(
            dirpath=f"train_logs/Unet++_resnet34_3class/checkpoints",
            # monitor="val_loss",
            # mode='min',
            filename='{epoch}-{valid_per_image_iou:.2f}-{val_loss:.2f}',
        )

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    logger = TensorBoardLogger("logs", name="BRTWModel",version ="Unet++_resnet34" )

    train_set = Dataset_3Building(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = True,pl =True)
    val_set = Dataset_3Building(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = False, pl = True)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=os.cpu_count())
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    
    model = BRTWModel("unetplusplus","resnet34", in_channels=3, out_classes= 4)
    # model.load_from_checkpoint(checkpoint_path="/mnt/data/bientd/segment_building/logs/BRTWModel/Unet++_resnet34/checkpoints/epoch=0-step=2295.ckpt")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=20,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        logger = logger,
        callbacks = [checkpoint_callback, VisualizeCallBack_fit(model,inference_epochs = 5)], 
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,

    )
    