from typing import Any, Optional, List

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from torch import Tensor
from torchmetrics.functional import f1, auroc


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 moco_extractor: Moco_v2,
                 num_classes: Optional[int] = 3,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 1e-4):
        super().__init__()
        self.backbone = moco_extractor.encoder_q
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.last_channel = self.backbone.fc.out_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx) -> Tensor:
        (img_q, img_k), y = batch
        output = self(img_q)
        loss = F.cross_entropy(output, y.long())
        acc1, acc5 = precision_at_k(output, y, top_k=(1, 3))
        f1_score = f1(output, y, num_classes=self.num_classes)

        log = {
            "train_loss": loss,
            "train_acc1": acc1,
            "train_acc5": acc5,
            "train_f1": f1_score,
        }
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_q, img_k), y = batch
        output = self(img_q)
        loss = F.cross_entropy(output, y.long())

        acc1, acc5 = precision_at_k(output, y, top_k=(1, 3))
        f1_score = f1(output, y, num_classes=self.num_classes)
        # auroc_score = auroc(F.softmax(output.detach()), y, num_classes=self.num_classes)
        return {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "val_f1": f1_score,
            # "val_auroc": auroc_score
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")
        val_f1 = mean(outputs, "val_f1")
        # val_auroc = mean(outputs, "val_auroc")
        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
            # "val_auroc": val_auroc
        }
        self.log_dict(log)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
