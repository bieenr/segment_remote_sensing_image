import os
import torch
# import pytorch_lightning as pl
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import torch.nn as nn
import pdb
from pprint import pprint
from pytorch_toolbelt import losses as L

from .jointloss import JointLoss

class BRTWModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        if out_classes == 1 :
            self.loss_fn  = JointLoss(L.DiceLoss("binary"),\
                                        #  L.SoftCrossEntropyLoss(reduction = "mean", ignore_index = -100, dim=1), 1.0, 1.0)
                                        nn.BCEWithLogitsLoss(),1.0,1.0)
            # smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True) 
        else :
            self.loss_fn =  JointLoss(L.DiceLoss("multilabel"),\
                                         nn.CrossEntropyLoss(), 1.0, 1.0)
            # smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True) 
        self.out_classes = out_classes
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.save_hyperparameters()
    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        if self.out_classes > 1:
            # pdb.set_trace()
            mask = F.one_hot(mask.long(), self.out_classes).permute(0,3,1,2).float()
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        if self.out_classes > 1 :
            pred_mask = prob_mask.argmax(dim=1)
            pred_mask =  F.one_hot(pred_mask.long(), self.out_classes).permute(0,3,1,2).float()
            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="multilabel", threshold=0.5)
        else :
            pred_mask = (prob_mask > 0.5).float()
            # We will compute IoU metric by two ways
            #   1. dataset-wise
            #   2. image-wise
            # but for now we just compute true positive, false positive, false negative and
            # true negative 'pixels' for each image and class
            # these values will be aggregated in the end of an epoch
            tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        output = {
            "loss": loss,
            "tp": tp.detach(),
            "fp": fp.detach(),
            "fn": fn.detach(),
            "tn": tn.detach(),
        }
        return output

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        pl.utilities.memory.garbage_collection_cuda()
        self.training_step_outputs.append(pl.utilities.memory.recursive_detach(output))
        self.log("train_loss", output["loss"], on_step=True, prog_bar=True)
        return output["loss"]

    def on_training_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        output = pl.utilities.memory.recursive_detach(self.shared_step(batch, "valid"))
        pl.utilities.memory.garbage_collection_cuda()
        self.validation_step_outputs.append(output)
        self.log("val_loss", output["loss"], on_step=True, prog_bar=True)
        return output["loss"]

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        output = pl.utilities.memory.recursive_detach(self.shared_step(batch, "test"))
        pl.utilities.memory.garbage_collection_cuda()
        self.test_step_outputs.append(output)
        return output["loss"]

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
if __name__ == "__main__":
    model = BRTWModel("FPN", "resnet34", in_channels=3, out_classes=5)
    x = torch.rand(3,512,512)
    print(model(x).shape)