from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.optimizers import Novograd
from pl_bolts.models.self_supervised import Moco_v2
from torch import Tensor
from torchmetrics import Accuracy, AUROC, Recall, Specificity
from torchmetrics.functional import auroc


# def filter_nans(logits, labels):
#     logits = logits[~torch.isnan(labels)]
#     labels = labels[~torch.isnan(labels)]
#     return logits, labels


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 val_pathology_list: List[str],
                 pretrained_checkpoint: str,
                 pos_weights: Optional = None,
                 in_channels: int = 3,
                 batch_size: int = 64,
                 learning_rate: float = 1e-3,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        assert Path(pretrained_checkpoint).expanduser().exists()
        moco_model = Moco_v2.load_from_checkpoint(pretrained_checkpoint)
        self.backbone = moco_model.encoder_q
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.last_channel = list(self.backbone.children())[-1].out_features
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

        self.train_acc = Accuracy(num_classes, average="none")
        self.train_sensitivity = Recall(num_classes, average="none")
        self.train_specificity = Specificity(num_classes, average="none")

        self.val_acc = [Accuracy(num_classes) for _ in val_pathology_list]
        self.val_sensitivity = Recall(num_classes, average="none")
        self.val_specificity = Specificity(num_classes, average="none")
        self.val_auroc = [AUROC(num_classes) for _ in val_pathology_list]

    def forward(self, x: Tensor) -> Tensor:
        embedding = self.backbone(x)
        return self.classifier(embedding)

    def training_step(self, batch, batch_idx) -> Tensor:
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, weight=self.pos_weights)
        self.log("train_loss", loss)
        output_acc = self.train_acc(output, target)

        for i, path in enumerate(self.hparams.val_pathology_list):
            self.log(f"train_acc_{path}", output_acc[i],
                     on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = F.cross_entropy(output, target, weight=self.pos_weights)

        result_logits = {}
        result_labels = {}
        self.log("val_loss", loss)
        self.val_sensitivity(output, target)
        self.val_specificity(output, target)
        for i, path in enumerate(self.hparams.val_pathology_list):
            logits, labels = output[:, i], target[:, i]
            result_logits[path] = logits
            result_labels[path] = labels

        return {"logits": result_logits, "targets": result_labels}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        auc_vals = []
        for i, path in enumerate(self.val_pathology_list):
            logits = []
            targets = []
            for output in outputs:
                logits.append(output["logits"][path].flatten())
                targets.append(output["targets"][path].flatten())

            logits = torch.cat(logits)
            targets = torch.cat(targets)

            self.val_acc[i](logits, targets)
            try:
                auc_val = auroc(torch.sigmoid(logits), targets)
                auc_vals.append(auc_val)
            except ValueError:
                auc_val = 0

            self.print(f"path: {path}, auc_val: {auc_val}")
            self.log(
                f"val_metrics/acc_{path}",
                self.val_acc[i],
                on_step=False,
                on_epoch=True,
            )
            self.log(f"val_auc_{path}", auc_val)

        self.log("val_auc_mean", sum(auc_vals) / len(auc_vals))

    def configure_optimizers(self):
        # optimizer = Novograd(self.parameters(), self.learning_rate)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     self.trainer.max_epochs,
        # )
        # return [optimizer], [scheduler]
        return Novograd([
            {"params": self.backbone.parameters(), "lr": 1e-5},
            {"params": self.classifier.parameters()}
        ], self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=3)
        parser.add_argument("--val_pathology_list", nargs="+")
        parser.add_argument("--pretrained_checkpoint", required=True)
        parser.add_argument("--pos_weights", nargs="+", type=float)
        parser.add_argument("--in_channels", default=3)
        parser.add_argument("--batch_size", default=64)
        parser.add_argument("--learning_rate", type=float, default=3e-2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--num_workers", type=int, default=4)

        return parser
