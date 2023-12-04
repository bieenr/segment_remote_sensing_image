from glob import glob
from timm.data import constants
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from tqdm import tqdm

from model.unet import UNet,BRTWModel
from callbacks.inference import Visualize_Image,  VisualizeImgFolder 
from data.BRTWDataset import BRTWDataset
### n_classes 
N_CLASSES = 5
####cuda
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###LOAD MODEL
CHECKPOINT_PATH = "/mnt/data/bientd/segment_building/train_logs/Building_non/Unet++_resnet34/checkpoints/epoch=0-valid_per_image_iou=0.95-val_loss=0.17.ckpt"

# MODEL = BRTWModel(n_channels=3, n_classes=N_CLASSES, bilinear=False)
# MODEL = BRTWModel("Unet","resnet34", in_channels=3, out_classes= 5)
# MODEL = MODEL.to(memory_format=torch.channels_last)
# state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
# MODEL.load_state_dict(state_dict['state_dict'], strict=False)
# MODEL.to(device=DEVICE)

MODEL = BRTWModel.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)


### Do output cua Segformer = (Size[0]/2) x (Size[1]/2) nen phai resize ve (Size[0] x Size[1]) 
if "Segformer" in str(type(MODEL)) or "segformer" in str(type(MODEL)):
    SCALE_FACTOR = 4
elif "Unet" in str(type(MODEL)) or "unet" in str(type(MODEL)):
    SCALE_FACTOR = 1 
else :
    SCALE_FACTOR = 1 
############ size anh
IMG_SIZE = (512,512)
############ path anh infer

PATH_IMAGE = "/mnt/data/RasterMask_v11/TrueOrtho/Ortho_Row(243)_Col(324).tif"
PATH_IMAGE_FOLDER = "/mnt/data/RasterMask_v11/"
############ save path

SAVE_PATH = "/mnt/data/bientd/segment_building/visualize/Building_non/Unet++_resnet34_f57291fab38e452e9c99b4a3f6e28274"
if(os.path.exists(SAVE_PATH) == False):
    os.mkdir(SAVE_PATH)
print("torch.cuda.is_available = ",torch.cuda.is_available())

if __name__ == '__main__':
    MODEL.eval()
    # VisualizeImgFolder(MODEL,PATH_IMAGE_FOLDER,IMG_SIZE,SAVE_PATH,scale_factor=SCALE_FACTOR,out_threshold=0.5,flag = 1)
    Visualize_Image(MODEL,PATH_IMAGE,IMG_SIZE,SAVE_PATH,scale_factor=1,out_threshold=0.5,flag = 1)