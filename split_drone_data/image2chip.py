import glob
import cv2
import numpy as np
import os
from tqdm import tqdm
import random

SAVE_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
IMAGE_FOLDER = "/mnt/data/RasterMask_v11/TrueOrtho"
# MASK_FOLDER = "/mnt/data/RasterMask_v11/Raster_BldgMask"
# MASK_FOLDER = "/mnt/data/RasterMask_v12/Mask2_Building"
MASK_FOLDER = "/mnt/data/RasterMask_v11/Mask2_Building"
TRAIN_TXT = "/mnt/data/RasterMask_v11/ImageSet/train.txt"
VAL_TXT = "/mnt/data/RasterMask_v11/ImageSet/val.txt"

os.makedirs(SAVE_FOLDER, exist_ok=True)
with open(TRAIN_TXT, "r") as f:
    TRAIN_LIST = f.readlines()
with open(VAL_TXT, "r") as f:
    VAL_LIST = f.readlines()

TRAIN_IMAGE_IDS = [x.strip()[6:].split(".")[0] for x in TRAIN_LIST]
VAL_IMAGE_IDS = [x.strip()[6:].split(".")[0] for x in VAL_LIST]

SIZE = 512
STRIDE = 512
AREA_THRESHOLD = 0.05


def check_empty_mask(mask):
    return np.all(mask == 0)


def check_area_mask(mask):
    num_non_zeros = np.count_nonzero(mask)
    W, H = mask.shape
    total_pixels = W * H
    ratio = num_non_zeros / total_pixels
    return (ratio < AREA_THRESHOLD) or (ratio > 1 - AREA_THRESHOLD)


def cut_a_image(id, mode):
    image_path = os.path.join(IMAGE_FOLDER, f"Ortho_{id}.tif")
    mask_path = os.path.join(MASK_FOLDER, f"Mask_Building_{id}.tif")
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path,0)
    W, H, C = image.shape
    assert image[:, :, 0].shape == mask.shape
    cols = np.math.floor((H - SIZE) / STRIDE + 1)
    rows = np.math.floor((W - SIZE) / STRIDE + 1)
    for col in range(cols):
        for row in range(rows):
            sub_id = f"{col}_{row}"
            sub_image = image[
                row * STRIDE : row * STRIDE + SIZE , col * STRIDE : col * STRIDE + SIZE, :
            ]
            sub_mask = mask[
                row * STRIDE : row * STRIDE + SIZE, col * STRIDE : col * STRIDE + SIZE
            ]

            save_image_path = os.path.join(
                # SAVE_FOLDER, mode, "images_raw", f"Ortho_{id}_{sub_id}.tif"
                SAVE_FOLDER, mode, "images_raw_v3-1", f"Ortho_{id}_{sub_id}.tif"
            )
            if random.randint(0, 9) > 4 and check_area_mask(sub_mask) and check_empty_mask(sub_mask):
                continue
            save_mask_path = os.path.join(
                # SAVE_FOLDER, mode, "masks_3class", f"Mask_{id}_{sub_id}.tif"
                SAVE_FOLDER, mode, "masks_buiding_v3-1", f"Mask_{id}_{sub_id}.tif"
            )

            cv2.imwrite(save_image_path, sub_image)
            cv2.imwrite(save_mask_path, sub_mask)


def cut_a_batch(ids, mode):
    os.makedirs(os.path.join(SAVE_FOLDER, mode),exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, mode, "images_raw_v3-1"),exist_ok=True)
    os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks_buiding_v3-1"),exist_ok=True)
    for id in tqdm(ids):
        cut_a_image(id, mode)


if __name__ == "__main__":
    cut_a_batch(TRAIN_IMAGE_IDS, "train")
    cut_a_batch(VAL_IMAGE_IDS, "val")
