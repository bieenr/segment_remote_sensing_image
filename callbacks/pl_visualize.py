from lightning.pytorch.callbacks import Callback
import torch 
import os
from tqdm import tqdm
from timm.data import constants
import torchvision.transforms as T
import cv2
import numpy as np
from clearml import Logger

TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(constants.IMAGENET_DEFAULT_MEAN, constants.IMAGENET_DEFAULT_STD),
    ]
)
value_colors = {1: (255,215,0), 2: (210,105,30), 3: (0, 255, 0), 4: (0,0,255)}

class VisualizeCallBack(Callback):
  def __init__(self, model, visualize_folder = "/mnt/data/RasterMask_v11/",inference_epochs = 5,size = (512,512) ) :
        super().__init__()
        self.model = model
        image_folder = os.path.join(visualize_folder, "TrueOrtho")
        with open(os.path.join(visualize_folder, "ImageSet", "test.txt"), "r") as f:
            image_names = f.readlines()
        self.img_paths = [os.path.join(image_folder, x.strip()) for x in image_names]
        self.size = size[0]
        self.inference_epochs = inference_epochs

  def on_train_end(self, trainer, pl_module):
    print("trainer.fit done")
    if (trainer.current_epoch + 1) % self.inference_epochs == 0:
        device = next(self.model.parameters()).device

        with torch.no_grad():
            self.model.eval()
            for path in tqdm(self.img_paths):
                img_name = path.split("/")[-1]
                # save_img_path = os.path.join(save_path, img_name)
                big_img = cv2.imread(path)
                big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
                total_scence = np.zeros_like(big_img)
                big_mask = big_img > 0
                # total_scence_reduce = np.mean(total_scence, axis=2)
                for x in range(0, big_img.shape[1] - self.size, self.size):
                    for y in range(0, big_img.shape[0] - self.size, self.size):
                        img = big_img[y : y + self.size, x : x + self.size, :]
                        tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
                        x_input = {'image':tensor_img}
                        # logits = pl_module(x_input["image"].to(device))
                        logits = self.model(x_input["image"].to(device))
                        

                        if logits.shape[1] == 1:
                            logits = logits.squeeze(0).squeeze(0)
                            logits = logits > 0
                            logits = logits.cpu().numpy().astype(np.uint8)
                        elif logits.shape[1] > 1:
                            logits = logits.argmax(dim=1)
                            logits = logits.squeeze().cpu().numpy()
                        masked_img = np.zeros_like(img)
                        for value, color in value_colors.items():
                            mask = (logits == value)
                            masked_img[mask] = color
                        colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
                        total_scence[y : y + self.size, x : x + self.size,:] = colored_mask_img
                total_scence = total_scence * big_mask
                total_scence = cv2.resize(total_scence, (1444, 1444))
                total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
                Logger.current_logger().report_image(
                    "Testing Image",
                    img_name,
                    # iteration=trainer.current_epoch,
                    image=total_scence,
                )



class VisualizeCallBack_fit(Callback):
  def __init__(self, model, visualize_folder = "/mnt/data/RasterMask_v11/",inference_epochs = 5,size = (512,512) ) :
        super().__init__()
        self.model = model
        image_folder = os.path.join(visualize_folder, "TrueOrtho")
        with open(os.path.join(visualize_folder, "ImageSet", "test.txt"), "r") as f:
            image_names = f.readlines()
        self.img_paths = [os.path.join(image_folder, x.strip()) for x in image_names]
        self.size = size[0]
        self.inference_epochs = inference_epochs

  def on_fit_end(self, trainer, pl_module):
    print("trainer.fit done")
    # if (trainer.current_epoch + 1) % self.inference_epochs == 0:
    device = next(self.model.parameters()).device

    with torch.no_grad():
        self.model.eval()
        for path in tqdm(self.img_paths):
            img_name = path.split("/")[-1]
            # save_img_path = os.path.join(save_path, img_name)
            big_img = cv2.imread(path)
            big_img = cv2.cvtColor(big_img, cv2.COLOR_BGR2RGB)
            total_scence = np.zeros_like(big_img)
            big_mask = big_img > 0
            # total_scence_reduce = np.mean(total_scence, axis=2)
            for x in range(0, big_img.shape[1] - self.size, self.size):
                for y in range(0, big_img.shape[0] - self.size, self.size):
                    img = big_img[y : y + self.size, x : x + self.size, :]
                    tensor_img = TRANSFORM(img).unsqueeze(0).cuda()
                    x_input = {'image':tensor_img}
                    # logits = pl_module(x_input["image"].to(device))
                    logits = self.model(x_input["image"].to(device))
                    

                    if logits.shape[1] == 1:
                        logits = logits.squeeze(0).squeeze(0)
                        logits = logits > 0
                        logits = logits.cpu().numpy().astype(np.uint8)
                    elif logits.shape[1] > 1:
                        logits = logits.argmax(dim=1)
                        logits = logits.squeeze().cpu().numpy()
                    masked_img = np.zeros_like(img)
                    for value, color in value_colors.items():
                        mask = (logits == value)
                        masked_img[mask] = color
                    colored_mask_img = cv2.addWeighted(img, 0.6, masked_img, 0.4, 0)
                    total_scence[y : y + self.size, x : x + self.size,:] = colored_mask_img
            total_scence = total_scence * big_mask
            total_scence = cv2.resize(total_scence, (1444, 1444))
            total_scence = cv2.cvtColor(total_scence, cv2.COLOR_BGR2RGB)
            Logger.current_logger().report_image(
                "Testing Image",
                img_name,
                # iteration=trainer.current_epoch,
                image=total_scence,
            )
