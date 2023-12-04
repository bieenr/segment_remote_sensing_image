from lightning.pytorch.callbacks import Callback
import torch 
import os
from tqdm import tqdm
from timm.data import constants
import torchvision.transforms as T
import cv2
import numpy as np
from clearml import Logger


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].long() + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class Eval_confusion_matrix(Callback):
    def __init__(self, model, loader ,out_classes = 4 ,size = (512,512)) :
            super().__init__()
            self.model = model
            self.size = size[0]
            self.loader = loader
            if out_classes > 1:
                self.confusion_matrix = runningScore(out_classes)
            else :
                self.confusion_matrix = runningScore(out_classes + 1)

    def on_test_end(self, trainer, pl_module):
        device = next(self.model.parameters()).device
        for batch in tqdm(self.loader):
            label_trues = batch["mask"].cpu()

            logits = self.model(batch["image"].to(device))
            if logits.shape[1] == 1:
                logits = logits.squeeze(0).squeeze(0)
                logits = logits > 0
                logits = logits.cpu().numpy().astype(np.uint8)
            elif logits.shape[1] > 1:
                logits = logits.argmax(dim=1)
                logits = logits.squeeze().cpu().numpy()

            self.confusion_matrix.update(label_trues, logits)
        
        total_pixel =  self.confusion_matrix.confusion_matrix.sum(0)
        print("Tong so pixel la :",total_pixel)
        print(np.round(self.confusion_matrix.confusion_matrix / total_pixel,5))
        print("-----------------------")
        print(self.confusion_matrix.confusion_matrix )
        # score , _ =  self.confusion_matrix.get_scores()
        # print(score)
        # print(_)
        self.confusion_matrix.reset()
