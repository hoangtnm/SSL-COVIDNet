from pathlib import Path
from typing import Any, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.optimizers import Novograd
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from torch import Tensor
from torchmetrics import Accuracy
from torchmetrics.functional import f1


def filter_nans(logits, labels):
    logits = logits[~torch.isnan(labels)]
    labels = labels[~torch.isnan(labels)]
    return logits, labels


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 moco_checkpoint: str,
                 num_classes: Optional[int] = 3,
                 batch_size: Optional[int] = 32,
                 learning_rate: Optional[float] = 1e-3,
                 epochs: Optional[int] = 5, **kwargs):
        super().__init__()
        assert Path(moco_checkpoint).expanduser().exists()
        moco_model = Moco_v2.load_from_checkpoint(moco_checkpoint)
        self.backbone = moco_model.encoder_q
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        self.last_channel = list(self.backbone.children())[-1].out_features
        self.classifier = nn.Sequential(
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(self.last_channel, num_classes)
        )
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx) -> Tensor:
        (img_q, img_k), y = batch
        output = self(img_q)
        loss = F.cross_entropy(output, y.long())
        acc1, acc5 = precision_at_k(output, y, top_k=(1, 3))
        # f1_score = f1(output, y, num_classes=self.num_classes)

        log = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
        }
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_q, img_k), y = batch
        output = self(img_q)
        loss = F.cross_entropy(output, y.long())

        acc1, acc5 = precision_at_k(output, y, top_k=(1, 3))
        _, pred = torch.max(output, 1)

        return {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "output": pred,
            "target": y,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")
        targets = torch.stack(
            [x["target"].detach() for x in outputs]).flatten()
        outputs = torch.stack(
            [x["output"].detach() for x in outputs]).flatten()

        val_f1 = f1(outputs, targets, num_classes=self.num_classes)
        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
        }
        self.log_dict(log)

    def configure_optimizers(self):
        # optimizer = Novograd(self.parameters(), self.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        # return [optimizer], [scheduler]
        return Novograd([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters()}
        ], self.learning_rate)
