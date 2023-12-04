import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from glob import glob
from timm.data import constants
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from tqdm import tqdm
from PIL import Image

from model.unet import UNet
# from mnt.data.luantranthanh.seg_building.model.Segformer import Segformer

TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
    ]
)
def Visualize(net,big_img,img_size,scale_factor=1,out_threshold=0.5):
    total_scence = np.zeros_like(big_img)
    big_mask = big_img > 0
    size = img_size[0]
    for x in range(0, big_img.shape[1] - size, size):
        for y in range(0, big_img.shape[0] - size, size):
            img = big_img[y : y + size, x : x + size, :]
            tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
            x_input = {'image':tensor_img}
            with torch.no_grad():
                mask_pred = net(tensor_img)
                # mask_pred = net(x_input)
            mask_pred = F.interpolate(
                mask_pred, scale_factor=scale_factor, mode="nearest"
            )
            if mask_pred.shape[1] == 1:
                mask_pred = mask_pred.squeeze(0).squeeze(0)
                mask_pred = mask_pred > 0
                mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
            elif mask_pred.shape[1] > 1:
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred = mask_pred.squeeze().cpu().numpy().astype(np.unit8)
            color = np.array([0, 255, 0], dtype="uint8")
            masked_img = np.where(mask_pred[..., None], color, img)

            colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
            total_scence[
                y : y + size, x : x + size, :
            ] = colored_mask_img
    total_scence = total_scence * big_mask
    total_scence = cv2.resize(total_scence, (1444, 1444))
    total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
    return total_scence

def load_image(path_folder):
    image_files = glob(path_folder + "*.tif")
    return image_files

PATH_IMAGE_FOLDER = "/mnt/data/RasterMask_v11/TrueOrtho/"
#### Layout
st.title("Inference Drone Image")
# uploaded_file = st.file_uploader("Choose a file", type = ["csv","png","jpg", "tif"])
image_files = load_image(PATH_IMAGE_FOLDER)
# st.multiselect("Select image: ",image_files)
image_path = st.selectbox("Select image: ",image_files,index =0 )
n_classes = st.number_input("Num classes: ",1)
Model_type = ( "Unet","Segformer" ) 
model_type = st.selectbox("SelectModel: ", Model_type, index =1 )

### n_classes 
N_CLASSES = n_classes 
####cuda
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###LOAD MODEL_UNET
if model_type == "Unet":
    CHECKPOINT_PATH_UNET = "/mnt/data/bientd/segment_building/checkpoints_unet/checkpoint_epoch20.pth"
    MODEL_UNET = UNet(n_channels=3, n_classes=N_CLASSES, bilinear=False)
    MODEL_UNET = MODEL_UNET.to(memory_format=torch.channels_last)
    state_dict = torch.load(CHECKPOINT_PATH_UNET, map_location=DEVICE)
    MODEL_UNET.load_state_dict(state_dict)
    MODEL_UNET.to(device=DEVICE)
    MODEL = MODEL_UNET
elif model_type == "Segformer":
    from argparse import Namespace
    args = Namespace()
    args.model = 'nvidia/segformer-b0-finetuned-ade-512-512'
    args.num_classes = 1
    args.look_around = False
    MODEL = Segformer(args)
    # print("error")
############ size anh for model predict
IMG_SIZE = (512,512)

col1, col2 = st.columns(2)
with col1 :
    st.header("raw")
    st.image(image_path)
big_img = Image.open(image_path)

big_img = cv2.cvtColor(np.array(big_img), cv2.COLOR_RGB2BGR)
predict_img = Visualize(MODEL,big_img,IMG_SIZE,scale_factor=1,out_threshold=0.5)
with col2 :
    st.header("predict")
    st.image(predict_img)
    

