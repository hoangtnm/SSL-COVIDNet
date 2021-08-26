from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from monai.optimizers import Novograd
from torch import Tensor
from torchmetrics import Accuracy, AUROC, Recall, Specificity

from .moco2_module import MoCoV2


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 val_pathology_list: List[str],
                 pretrained_checkpoint: str,
                 pos_weights: Optional = None,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = False,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        assert num_classes == len(val_pathology_list)
        assert Path(pretrained_checkpoint).expanduser().exists()
        self.save_hyperparameters()
        moco_model = MoCoV2.load_from_checkpoint(pretrained_checkpoint)
        self.base_encoder = moco_model.hparams.base_encoder
        self.in_channels = moco_model.hparams.in_channels
        self.backbone = moco_model.encoder_q
        for param in self.backbone.parameters():
            param.requires_grad = False

        try:
            self.last_channel = list(self.backbone.children())[-1].out_features
        except AttributeError:
            self.last_channel = list(self.backbone.children())[
                -1].out.out_features

        self.classifier = nn.Sequential(
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(self.last_channel, num_classes)
        )
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if pos_weights is None:
            pos_weights = torch.ones(num_classes)
        self.register_buffer("pos_weights", pos_weights)

        self.train_acc = Accuracy(num_classes=num_classes,
                                  average="none")
        # self.train_sensitivity = Recall(num_classes=num_classes,
        #                                 average="none")
        # self.train_specificity = Specificity(num_classes=num_classes,
        #                                      average="none")

        self.val_acc = Accuracy(num_classes=num_classes,
                                average="none")
        self.val_sensitivity = Recall(num_classes=num_classes,
                                      average="none")
        self.val_specificity = Specificity(num_classes=num_classes,
                                           average="none")
        self.val_auc = AUROC(num_classes=num_classes,
                             average=None)

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx) -> Tensor:
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, weight=self.pos_weights)
        output_acc = self.train_acc(output, target)

        for i, path in enumerate(self.hparams.val_pathology_list):
            self.log(f"train_acc_{path}", output_acc[i])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, weight=self.pos_weights)

        self.val_acc(output, target)
        self.val_sensitivity(output, target)
        self.val_specificity(output, target)
        self.val_auc(output, target)
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_acc = self.val_acc.compute()
        val_sensitivity = self.val_sensitivity.compute()
        val_specificity = self.val_specificity.compute()
        val_auc = self.val_auc.compute()
        for i, path in enumerate(self.hparams.val_pathology_list):
            log = ({
                f"val_acc/{path}": val_acc[i],
                f"val_sensitivity/{path}": val_sensitivity[i],
                f"val_specificity/{path}": val_specificity[i],
                f"val_auc/{path}": val_auc[i],
            })
            self.log_dict(log)
            self.print({k: v.item() for k, v in log.items()})
        self.log("val_auc_mean", torch.mean(val_auc))
        self.log("hp_metric", torch.mean(val_auc))

    def configure_optimizers(self):
        optimizer = Novograd([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters()}
        ], self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs,
        )

        if self.hparams.use_lr_scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_dir")
        parser.add_argument("--num_classes", type=int, default=3)
        parser.add_argument("--val_pathology_list", nargs="+",
                            default=["normal", "pneumonia", "covid"])
        parser.add_argument("--pretrained_checkpoint", required=True)
        parser.add_argument("--pos_weights", nargs="+", type=float)
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--use_lr_scheduler", action="store_true")

        return parser
