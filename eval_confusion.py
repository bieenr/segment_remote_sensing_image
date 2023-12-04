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

from data.BRTWDataset import BRTWDataset, Dataset_3Building
from model.unet import BRTWModel, PlModel, UNet
from callbacks.pl_visualize import VisualizeCallBack,VisualizeCallBack_fit
from callbacks.eval_confusion_matrix import Eval_confusion_matrix

# task = Task.init(
#     project_name="Getting Started",
#     task_name="BRTWModel_Unet++_resnet34 Training",
# )

ROOT_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
IMG_SIZE = (512,512)
BATCH_SIZE = 4
CLASSES = 5

if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')

    logger = TensorBoardLogger("logs", name="BRTWModel",version ="Unet++_resnet34" )

    val_set = Dataset_3Building(root_folder = ROOT_FOLDER, img_size = IMG_SIZE, is_training = False, pl = True)
    loader_args = dict(batch_size=BATCH_SIZE, num_workers=os.cpu_count())

    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    model = BRTWModel.load_from_checkpoint(checkpoint_path="/mnt/data/bientd/segment_building/train_logs/Unet++_resnet34_3class/checkpoints/epoch=19-valid_per_image_iou=0.90-val_loss=0.42.ckpt")

    _, test_set = torch.utils.data.random_split(val_set,[0.99,0.01])
    test_loader = DataLoader(test_set, shuffle=False, **loader_args)

    trainer = pl.Trainer(
        # accelerator="gpu",
        max_epochs=1,
        logger = logger,
        callbacks = [ Eval_confusion_matrix(model,val_loader,out_classes=4)], 
    )
    trainer.test(model, dataloaders=test_loader, verbose=False)