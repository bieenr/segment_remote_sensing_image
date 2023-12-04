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
from data.BuildingDataset import BuildingDataset
from data.BRTWDataset import BRTWDataset
from model.unet import BRTWModel, PlModel, UNet
from callbacks.pl_visualize import VisualizeCallBack,VisualizeCallBack_fit

task = Task.init(
    project_name="Getting Started",
    task_name="Build_non_Model_Unet++_resnet34 Training",
    # output_uri=True  # IMPORTANT: setting this to True will upload the model
    # If not set the local path of the model will be saved instead!
)


ROOT_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
IMG_SIZE = (512,512)
BATCH_SIZE = 4
CLASSES = 5

checkpoint_callback = ModelCheckpoint(
            dirpath=f"train_logs/Building_non/Unet++_resnet34/checkpoints",
            monitor="val_loss",
            mode='min',
            filename='{epoch}-{valid_per_image_iou:.2f}-{val_loss:.2f}',
        )

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    logger = TensorBoardLogger("logs", name="Building_non_Model",version ="Unet++_resnet34" )

    train_set = BuildingDataset(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = True,pl =True)
    val_set = BuildingDataset(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = False,pl =True)
    # train_set = BRTWDataset(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = True,pl =True)
    # val_set = BRTWDataset(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = False, pl = True)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=os.cpu_count())
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    DEVICE =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = BRTWModel("Unet","resnet34", in_channels=3, out_classes= 5)
    # model = model.to(memory_format=torch.channels_last)
    # state_dict = torch.load("/mnt/data/bientd/segment_building/train_logs/Unet/checkpoints/best.ckpt", map_location=DEVICE)
    # model.load_state_dict(state_dict['state_dict'], strict=False)
    # model.to(device=DEVICE)

    # /mnt/data/bientd/segment_building/train_logs/Unet++_resnet34/checkpoints/epoch=19-valid_per_image_iou=0.83-val_loss=0.44.ckpt
    # model = BRTWModel("unetplusplus","resnet34", in_channels=3, out_classes= 2)
    model = BRTWModel.load_from_checkpoint(checkpoint_path="/mnt/data/bientd/segment_building/train_logs/Building_non/Unet++_resnet34/checkpoints/epoch=0-valid_per_image_iou=0.95-val_loss=0.17.ckpt")
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=0,
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        logger = logger,
        callbacks = [VisualizeCallBack_fit(model,inference_epochs = 5),checkpoint_callback]
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,

    )

    # run validation dataset
    valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
    pprint(valid_metrics)
    # trainer.save_checkpoint(os.path.join(f"train_logs/Building_non/Unet++_resnet34/checkpoints","lastest.ckpt"))