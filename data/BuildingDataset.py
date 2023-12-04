from torch.utils.data import Dataset
import torch
import os
from glob import glob
import cv2
import torchvision.transforms as T
from timm.data import constants
import albumentations as A
import numpy as np
class BuildingDataset(Dataset):
    def __init__(self,root_folder :str, img_size: tuple, is_training :bool ,pl : False):
        mode = "train" if is_training else "val"
        self.root_folder = root_folder
        self.pl = pl 
        
        self.image_folder = os.path.join(root_folder, mode, "images_raw_v3-1")
        self.mask_folder = os.path.join(root_folder, mode, "masks_buiding_v3-1")
        # self.image_paths = glob(self.image_folder + "/*.tif")
        self.image_paths = glob(self.mask_folder + "/*.tif")

        self.img_size = img_size
        self.is_training = is_training
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD
                ),
            ]
        )
        self.augment = A.Compose(
            [
                A.RandomResizedCrop(
                    height = img_size[0],
                    width = img_size[1],
                    scale = (0.75,1),
                    always_apply = True,
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(),
            ]   
        )
    def __len__(self):
        return len(self.image_paths)
    def resize(self, img, size):
        return cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    def __getitem__(self,index):
        mask_path = self.image_paths[index]
        img_name = mask_path.split("/")[-1].replace("Mask", "Ortho")
        img_path = os.path.join(self.image_folder,img_name)
        # img_path = self.image_paths[index]
        # mask_name = img_path.split("/")[-1].replace("Ortho", "Mask")
        # mask_path = os.path.join(self.mask_folder,mask_name)
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        if self.is_training :
            augmented = self.augment(image = img, mask = mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = self.resize(img,self.img_size)
        # mask = self.resize(mask,(self.img_size[0]//4, self.img_size[0]//4)) # model segformer
        mask = self.resize(mask,(self.img_size[0], self.img_size[0]))
        mask = mask > 0
        img = self.transform(img)
        # mask = torch.tensor(mask).float().unsqueeze(0)  # unsqueeze(0) for segformer
        mask = torch.tensor(mask).float()
        if self.pl :
            return {"image" :img, "mask" :mask}
        return img, mask

if __name__ == "__main__":

    root_folder = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
    img_size = (512,512)
    dataset = BuildingDataset(
        root_folder = root_folder, img_size = img_size, is_training = True
    )
    img, mask = dataset[0]
    print(img.shape, mask.shape)
    # print(torch.unique(mask))