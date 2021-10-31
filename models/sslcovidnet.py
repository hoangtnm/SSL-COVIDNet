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


# class FocalLoss(nn.Module):
#     def __init__(self, alpha: float = 0, gamma: float = 0,
#                  size_average: bool = True):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.size_average = size_average
#
#     def forward(self, input: Tensor, target: Tensor):
#         pass


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 pretrained: str,
                 num_classes: int = 3,
                 pos_weights: Optional[List[float]] = None,
                 learning_rate: float = 1e-3,
                 use_lr_scheduler: bool = False,
                 data_dir: str = "./",
                 batch_size: int = 128,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.pathologies = ["normal", "pneumonia", "covid"]
        backbone = MoCoV2.load_from_checkpoint(pretrained)
        self.encoder = backbone.encoder_q
        if backbone.hparams.use_mlp:
            if hasattr(self.encoder, "fc"):  # ResNet models
                dim_mlp = self.encoder.fc[-1].weight.shape[1]
                self.encoder.fc = nn.Linear(dim_mlp, num_classes)
            elif hasattr(self.encoder, "classifier"):  # Densenet models
                dim_mlp = self.encoder.classifier[-1].weight.shape[1]
                self.encoder.classifier = nn.Linear(dim_mlp, num_classes)

        elif hasattr(self.encoder, "fc"):  # ResNet models
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Linear(dim_mlp, num_classes)
        elif hasattr(self.encoder, "classifier"):  # Densenet models
            dim_mlp = self.encoder.classifier.weight.shape[1]
            self.encoder.classifier = nn.Linear(dim_mlp, num_classes)

        for name, param in self.encoder.named_parameters():
            if name not in ("fc.weight", "fc.bias",
                            "classifier.weight", "classifier.bias"):
                param.requires_grad = False

        pos_weights = torch.tensor(pos_weights) if pos_weights \
            else torch.ones(num_classes)
        self.register_buffer("pos_weights", pos_weights)
        self.criterion = nn.CrossEntropyLoss(weight=self.pos_weights,
                                             label_smoothing=0.15)

        self.train_acc = Accuracy(num_classes=num_classes,
                                  average="none")
        # self.train_sensitivity = Recall(num_classes=num_classes,
        #                                 average="none")
        # self.train_specificity = Specificity(num_classes=num_classes,
        #                                      average="none")

        self.val_acc = Accuracy(num_classes=num_classes, average="none")
        self.val_sensitivity = Recall(num_classes=num_classes, average="none")
        self.val_specificity = Specificity(num_classes=num_classes,
                                           average="none")
        self.val_auc = AUROC(num_classes=num_classes, average=None)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        inputs, target = batch
        output = self(inputs)
        loss = self.criterion(output, target)
        output_acc = self.train_acc(output, target)

        for i, path in enumerate(self.pathologies):
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
        val_auc_mean = torch.mean(val_auc)
        for i, path in enumerate(self.pathologies):
            log = {
                f"val_acc/{path}": val_acc[i],
                f"val_sensitivity/{path}": val_sensitivity[i],
                f"val_specificity/{path}": val_specificity[i],
                f"val_auc/{path}": val_auc[i],
            }
            self.log_dict(log)
            self.print({k: v.item() for k, v in log.items()})
        self.log_dict({
            "hp_metric": val_auc_mean,
            "val_auc_mean": val_auc_mean,
        })

    def configure_optimizers(self):
        optimizer = Novograd(self.parameters(), self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs,
        )

        if self.hparams.use_lr_scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer
