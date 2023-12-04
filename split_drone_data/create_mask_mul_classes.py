"""
create mask multiple with 0-background 1-building 2-road 3-tree 4-water
"""

import glob
import os
import cv2 
import numpy as np
from tqdm import tqdm

ROOT_FOLDER = "/mnt/data/bientd/segment_building/dataset/chips-512-full"
CLASSES = ["masks_bldg","masks_road","masks_tree","masks_water"]
CLASSES_SPLIT = ["masks_bldg/BldgMask","masks_road/RoadMask","masks_tree/TreeMask","masks_water/WaterMask"]
SIZE = 512
STRIDE = 512
AREA_THRESHOLD = 0.1
VISUAL_PATH = "masks_multiple_visual"
TRAIN__PATH = "masks_multiple"
"""
This function create mask by stack multiple mask  order bldg-road-tree-water (water -highest priority)
"""
def create_mask(ls):
    bldg_mask = cv2.imread(ls[0],0)   
    road_mask = cv2.imread(ls[1],0)
    tree_mask = cv2.imread(ls[2],0)
    water_mask = cv2.imread(ls[3],0)
    assert bldg_mask.shape == road_mask.shape and road_mask.shape == tree_mask.shape and tree_mask.shape == water_mask.shape
    mask = np.zeros_like(bldg_mask)
    mask[bldg_mask > 0] = 1#64
    mask[road_mask > 0] = 2#128
    mask[tree_mask > 0] = 3#192
    mask[water_mask > 0] = 4#255
    save_path = ls[4]

    if not check_area_mask(mask):
        cv2.imwrite(save_path,mask)
    return 


def check_empty_mask(mask):
    return np.all(mask == 0)


def check_area_mask(mask):
    num_bldg = np.count_nonzero(mask == 1)
    num_road = np.count_nonzero(mask == 2)
    num_tree = np.count_nonzero(mask == 3)
    num_water = np.count_nonzero(mask == 4)
    W, H = mask.shape
    total_pixels = W * H
    ratio_bldg = num_bldg / total_pixels
    ratio_water = num_water / total_pixels
    ratio_tree = num_tree / total_pixels
    ratio_road = num_road / total_pixels
    return ((ratio_bldg < AREA_THRESHOLD)  or (ratio_bldg > 1 - AREA_THRESHOLD)) and (ratio_tree < 0.01) \
             and (ratio_water < 0.01) and (ratio_road < AREA_THRESHOLD)

def convert_bldg_path(path):
    ls = path.split("masks_bldg/BldgMask")
    ls_path = []
    for i in CLASSES_SPLIT:
        path_new =ls[0] + i + ls[1]
        ls_path.append(path_new)
    ls_path.append(os.path.join(ls[0],"masks_multiple","Mask"+ls[1]))
    return ls_path


if __name__ == "__main__":
    print("start")
    mode = "val"
    save_folder = "/mnt/data/bientd/segment_building/dataset/chips-512-full/" + mode + "/masks_multiple"
    if not os.path.exists(save_folder) :
        os.makedirs(save_folder)
    list_classes = glob.glob(os.path.join(ROOT_FOLDER,mode,CLASSES[0])+"/*.tif")
    for ls in tqdm(list_classes):
        ls = convert_bldg_path(ls)
        create_mask(ls)

    print(len(glob.glob(save_folder+"/*.tif")))