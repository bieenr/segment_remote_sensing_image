import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# import wandb
from eval.evaluate import evaluate
from model.unet import UNet_beta
from data.BuildingDataset import BuildingDataset
from utils.dice_score import dice_loss


if __name__ == '__main__':
    dir_checkpoint = Path('/mnt/data/bientd/segment_building/checkpoints_unet/')
    root_folder = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
    img_size = (512,512)
    batch_size = 4
    print("torch.cuda.is_available = ",torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet_beta(n_channels=3, n_classes= 1 , bilinear=False)
    model = model.to(memory_format=torch.channels_last)


    # path_load_model = "/mnt/data/bientd/segment_building/checkpoints_unet/checkpoint_beta_epoch33.pth"
    # state_dict = torch.load(path_load_model, map_location=device)
    # model.load_state_dict(state_dict)

    model.to(device=device)
    train_set = BuildingDataset(root_folder = root_folder, img_size = img_size, is_training = True)
    train_set, val_set_1 = random_split(train_set, [int(len(train_set)*0.25), len(train_set)-int(len(train_set)*0.25)], generator=torch.Generator().manual_seed(0))
    model(train_set[0][0].unsqueeze(0).to(device))
