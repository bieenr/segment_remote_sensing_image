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
def Visualize_Image(net,path,img_size,save_path,scale_factor=1,out_threshold=0.5):
    save_img_path = os.path.join(save_path, path.split("/")[-1]).split(".")[0]
    # print(save_img_path)
    big_img = cv2.imread(path)
    big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
    total_scence = np.zeros_like(big_img)
    big_mask = big_img > 0
    size = img_size[0]
    for x in range(0, big_img.shape[1] - size, size):
        for y in range(0, big_img.shape[0] - size, size):
            img = big_img[y : y + size, x : x + size, :]
            tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
            x_input = {'image':tensor_img}
            # if self.args.look_around:
            #     for d in [[1,0]]:
            #         _x = x + d[0]*size
            #         _y = y + d[1]*size
            #         _img = big_img[_y : _y + size, _x : _x + size, :]
            #         _img = TRANSFORM(_img).cuda().unsqueeze(0)
            #     x_input['around'] = _img
            with torch.no_grad():
                mask_pred = net(tensor_img)
                # mask_pred = net(x_input)
            mask_pred = F.interpolate(
                mask_pred, scale_factor=scale_factor, mode="nearest"
            )
            mask_pred = mask_pred.squeeze(0).squeeze(0)
            mask_pred = mask_pred > 0
            mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
            color = np.array([0, 255, 0], dtype="uint8")
            masked_img = np.where(mask_pred[..., None], color, img)
            colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
            total_scence[
                y : y + size, x : x + size, :
            ] = colored_mask_img
    total_scence = total_scence * big_mask
    total_scence = cv2.resize(total_scence, (1444, 1444))
    total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{save_img_path}_predict.jpg', total_scence)


def VisualizeImgFolder(net,visualize_input,img_size,save_path,scale_factor=1,out_threshold=0.5):#device
  image_folder = os.path.join(visualize_input, "TrueOrtho")
  with open(os.path.join(visualize_input,"ImageSet", "test.txt"),"r") as f:
    image_names = f.readlines()
  img_paths = [os.path.join(image_folder, x.strip()) for x in image_names]
  size = img_size[0]
  for path in tqdm(img_paths):
    img_name = path.split("/")[-1]
    save_img_path = os.path.join(save_path, img_name)
    big_img = cv2.imread(path)
    big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
    total_scence = np.zeros_like(big_img)
    big_mask = big_img > 0
    for x in range(0, big_img.shape[1] - size, size):
        for y in range(0, big_img.shape[0] - size, size):
            img = big_img[y : y + size, x : x + size, :]
            tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
            x_input = {'image':tensor_img}
            # if self.args.look_around:
            #     for d in [[1,0]]:
            #         _x = x + d[0]*size
            #         _y = y + d[1]*size
            #         _img = big_img[_y : _y + size, _x : _x + size, :]
            #         _img = TRANSFORM(_img).cuda().unsqueeze(0)
            #     x_input['around'] = _img
            with torch.no_grad():
                mask_pred = net(tensor_img)
                # mask_pred = net(x_input)
            mask_pred = F.interpolate(
                mask_pred, scale_factor=scale_factor, mode="nearest"
            )
            mask_pred = mask_pred.squeeze(0).squeeze(0)
            mask_pred = mask_pred > 0
            mask_pred = mask_pred.cpu().numpy().astype(np.uint8)
            color = np.array([0, 255, 0], dtype="uint8")
            masked_img = np.where(mask_pred[..., None], color, img)
            colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
            total_scence[
                y : y + size, x : x + size, :
            ] = colored_mask_img
    total_scence = total_scence * big_mask
    total_scence = cv2.resize(total_scence, (1444, 1444))
    total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f'{img_name}_predict.jpg', total_scence)

if __name__ == '__main__':
    checkpoint_path = "/mnt/data/bientd/segment_building/checkpoints_unet/checkpoint_epoch20.pth"
    img_size = (512,512)
    # path = "/mnt/data/RasterMask_v11/TrueOrtho/Ortho_Row(252)_Col(316).tif"
    path = "/mnt/data/RasterMask_v11/"
    save_path = "/mnt/data/bientd/segment_building/visualize/"
    if(os.path.exists(save_path) == False):
        os.mkdir(save_path)
    print("torch.cuda.is_available = ",torch.cuda.is_available())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = model.to(memory_format=torch.channels_last)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device=device)
    VisualizeImgFolder(model,path,img_size,save_path,scale_factor=1,out_threshold=0.5)