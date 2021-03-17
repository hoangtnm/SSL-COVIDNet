import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from pl_bolts.models.self_supervised import MocoV2
from pl_bolts.metrics import mean, precision_at_k
from typing import Any, Optional


class MocoCOVIDNet(pl.LightningModule):
    def __init__(self,
                 moco_extractor: MocoV2,
                 num_classes: int = 3,
                 batch_size: int = 32,
                 lr: int = 1e-3):
        super().__init__()
        self.backbone = moco_extractor.encoder_q
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.last_channel = self.backbone.fc.out_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )
        self.batch_size = batch_size
        self.learning_rate = lr

    def forward(self, x):
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y.long())
        acc1, acc5 = precision_at_k(output, y, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y.long())

        acc1, acc5 = precision_at_k(output, y, top_k=(1, 5))
        return {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}

    def validation_epoch_end(self, outputs) -> None:
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")
        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
