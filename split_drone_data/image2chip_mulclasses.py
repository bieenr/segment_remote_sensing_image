import glob
import cv2
import numpy as np
import os
from tqdm import tqdm

SAVE_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
# IMAGE_FOLDER = "/mnt/data/RasterMask_v11/TrueOrtho"
MASK_BLDG_FOLDER = "/mnt/data/RasterMask_v11/Raster_BldgMask"
MASK_WATER_FOLDER = "/mnt/data/RasterMask_v11/Raster_WaterMask"
MASK_TREE_FOLDER = "/mnt/data/RasterMask_v11/Raster_TreeMask"
MASK_ROAD_FOLDER = "/mnt/data/RasterMask_v11/Raster_RoadMask"
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
    mask_bldg_path = os.path.join(MASK_BLDG_FOLDER, f"BldgMask_{id}.tif")
    mask_tree_path = os.path.join(MASK_TREE_FOLDER, f"TreeMask_{id}.tif")
    mask_road_path = os.path.join(MASK_ROAD_FOLDER, f"RoadMask_{id}.tif")
    mask_water_path = os.path.join(MASK_WATER_FOLDER, f"WaterMask_{id}.tif")

    mask_bldg = cv2.imread(mask_bldg_path, 0)
    mask_tree = cv2.imread(mask_tree_path, 0)
    mask_road = cv2.imread(mask_road_path, 0)
    mask_water = cv2.imread(mask_water_path, 0)
    W, H = mask_bldg.shape
    # assert image[:, :, 0].shape == mask.shape
    cols = np.math.floor((H - SIZE) / STRIDE + 1)
    rows = np.math.floor((W - SIZE) / STRIDE + 1)
    for col in range(cols):
        for row in range(rows):
            sub_id = f"{col}_{row}"
            sub_mask_bldg = mask_bldg[
                row * STRIDE : row * STRIDE + SIZE, col * STRIDE : col * STRIDE + SIZE
            ]
            sub_mask_road = mask_road[
                row * STRIDE : row * STRIDE + SIZE, col * STRIDE : col * STRIDE + SIZE
            ]
            sub_mask_tree = mask_tree[
                row * STRIDE : row * STRIDE + SIZE, col * STRIDE : col * STRIDE + SIZE
            ]
            sub_mask_water = mask_water[
                row * STRIDE : row * STRIDE + SIZE, col * STRIDE : col * STRIDE + SIZE
            ]
            # if check_empty_mask(sub_mask_bldg):
            #     continue

            save_mask_bldg_path = os.path.join(
                SAVE_FOLDER, mode, "masks_bldg", f"BldgMask_{id}_{sub_id}.tif"
            )
            cv2.imwrite(save_mask_bldg_path, sub_mask_bldg)

            save_mask_road_path = os.path.join(
                SAVE_FOLDER, mode, "masks_road", f"RoadMask_{id}_{sub_id}.tif"
            )
            cv2.imwrite(save_mask_road_path, sub_mask_road)

            save_mask_tree_path = os.path.join(
                SAVE_FOLDER, mode, "masks_tree", f"TreeMask_{id}_{sub_id}.tif"
            )
            cv2.imwrite(save_mask_tree_path, sub_mask_tree)

            save_mask_water_path = os.path.join(
                SAVE_FOLDER, mode, "masks_water", f"WaterMask_{id}_{sub_id}.tif"
            )
            cv2.imwrite(save_mask_water_path, sub_mask_water)

def cut_a_batch(ids, mode):
    if not os.path.exists(os.path.join(SAVE_FOLDER, mode)):
        os.makedirs(os.path.join(SAVE_FOLDER, mode))
    if not os.path.exists(os.path.join(SAVE_FOLDER, mode, "masks_bldg")):
        os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks_bldg"))
    if not os.path.exists(os.path.join(SAVE_FOLDER, mode, "masks_road")) :
        os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks_road"))
    if not os.path.exists(os.path.join(SAVE_FOLDER, mode, "masks_tree")) :   
        os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks_tree"))
    if not os.path.exists(os.path.join(SAVE_FOLDER, mode, "masks_water")):
        os.makedirs(os.path.join(SAVE_FOLDER, mode, "masks_water"))
    for id in tqdm(ids):
        cut_a_image(id, mode)


if __name__ == "__main__":
    cut_a_batch(TRAIN_IMAGE_IDS, "train")
    cut_a_batch(VAL_IMAGE_IDS, "val")
