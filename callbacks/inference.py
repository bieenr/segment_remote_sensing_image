from glob import glob
from timm.data import constants
import torchvision.transforms as T
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import os
from tqdm import tqdm

from model.unet import UNet

TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
    ]
)


def Visualize_Image(
    net, path, img_size, save_path, scale_factor=1, out_threshold=0.5, flag=None
):
    save_img_path = os.path.join(save_path, path.split("/")[-1]).split(".")[0]
    big_img = cv2.imread(path)
    big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
    total_scence = np.zeros_like(big_img)
    total_scence_reduce = np.mean(total_scence, axis=2)
    # print(total_scence_reduce.shape)
    big_mask = big_img > 0
    size = img_size[0]

    for x in range(0, big_img.shape[1], size):
        for y in range(0, big_img.shape[0], size):
            img = np.zeros((size, size, big_img.shape[2]), dtype=np.uint8)
            img[: img.shape[0], : img.shape[1], :] = big_img[
                y : y + size, x : x + size, :
            ]
            if torch.cuda.is_available():
                tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
                net = net.cuda()
            else:
                tensor_img = TRANSFORM(img).unsqueeze(0)
            x_input = {"image": tensor_img}

            mask_pred = net(tensor_img)

            if mask_pred.shape[1] == 1:
                mask_pred = mask_pred.squeeze(0).squeeze(0)
                mask_pred = mask_pred > 0
                mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
                mask_pred[mask_pred == 0] = 0
                mask_pred[mask_pred == 1] = 255

            elif mask_pred.shape[1] > 1:
                mask_pred = mask_pred.argmax(dim=1)
                mask_pred = mask_pred.squeeze().cpu().numpy()  # .astype(np.unit8)
                # visualize image
                if flag == None:
                    mask_pred[mask_pred == 1] = 64
                    mask_pred[mask_pred == 2] = 128
                    mask_pred[mask_pred == 3] = 191
                    mask_pred[mask_pred == 4] = 255
                elif flag == -1:
                    mask_pred[mask_pred != 0] = 255
                    mask_pred[mask_pred == 0] = 0
                elif flag != -1 and flag != None:
                    mask_pred[mask_pred != flag] = 0
                    mask_pred[mask_pred == flag] = 255
            total_scence_reduce[y : y + size, x : x + size] = mask_pred
    total_scence_reduce = total_scence_reduce * big_mask[:, :, 0]
    total_scence_reduce = cv2.resize(total_scence_reduce, (1444, 1444))
    # print(total_scence_reduce[10000:10100, 10000:10100])
    cv2.imwrite(f"{save_img_path}_predict.tif", total_scence_reduce)


def VisualizeImgFolder(
    net,
    visualize_input,
    img_size,
    save_path,
    scale_factor=1,
    out_threshold=0.5,
    flag=None,
):  # device
    image_folder = os.path.join(visualize_input, "TrueOrtho")
    with open(os.path.join(visualize_input, "ImageSet", "test.txt"), "r") as f:
        image_names = f.readlines()
    img_paths = [os.path.join(image_folder, x.strip()) for x in image_names]

    size = img_size[0]
    for path in tqdm(img_paths):
        img_name = path.split("/")[-1].split(".")[0]
        save_img_path = os.path.join(save_path, img_name)

        big_img = cv2.imread(path)
        big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
        total_scence_reduce = np.zeros_like(big_img[:, :, 0])
       
        # total_scence_reduce = np.mean(total_scence, axis=2)
        big_mask = big_img > 0
        for x in range(0, big_img.shape[1], size):
            for y in range(0, big_img.shape[0], size):
                img = np.zeros((size, size, big_img.shape[2]), dtype=np.uint8)
                img[: img.shape[0], : img.shape[1], :] = big_img[
                    y : y + size, x : x + size, :
                ]
                if torch.cuda.is_available():
                    tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
                    net = net.cuda()
                else:
                    tensor_img = TRANSFORM(img).unsqueeze(0)
                x_input = {"image": tensor_img}

                with torch.no_grad():
                    mask_pred = net(tensor_img)

                # mask_pred = F.interpolate(
                #     mask_pred, scale_factor=scale_factor, mode="nearest"
                # )
                if mask_pred.shape[1] == 1:
                    mask_pred = mask_pred.squeeze(0).squeeze(0)
                    mask_pred = mask_pred > 0
                    mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
                    mask_pred[mask_pred == 0] = 0
                    mask_pred[mask_pred == 1] = 255
                elif mask_pred.shape[1] > 1:
                    mask_pred = mask_pred.argmax(dim=1)
                    mask_pred = mask_pred.squeeze().cpu().numpy()  # .astype(np.unit8)
                    # visualize image
                    if flag == None:
                        mask_pred[mask_pred == 1] = 64
                        mask_pred[mask_pred == 2] = 128
                        mask_pred[mask_pred == 3] = 191
                        mask_pred[mask_pred == 4] = 255
                    elif flag == -1:
                        mask_pred[mask_pred != 0] = 255
                        mask_pred[mask_pred == 0] = 0
                    elif flag != -1 and flag != None:
                        mask_pred[mask_pred != flag] = 0
                        mask_pred[mask_pred == flag] = 255
                total_scence_reduce[y : y + size, x : x + size] = mask_pred
        total_scence_reduce = total_scence_reduce * big_mask[:, :, 0]
        total_scence_reduce = cv2.resize(total_scence_reduce, (1444, 1444))

        cv2.imwrite(f"{save_img_path}_predict.tif", total_scence_reduce)
