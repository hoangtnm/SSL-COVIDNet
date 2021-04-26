from typing import Any, Optional, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.models.self_supervised import Moco_v2
from torch import Tensor
from torchmetrics.functional import f1

from datasets.covidxct import SSLCOVIDxCTIterator, ssl_covidxct_train_pipeline, \
    ssl_covidxct_val_pipeline
from utils.dali import DALIGenericIteratorV2


class SSLCOVIDNet(pl.LightningModule):
    def __init__(self,
                 moco_extractor: Moco_v2,
                 num_classes: Optional[int] = 3,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 1e-3):
        super().__init__()
        self.backbone = moco_extractor.encoder_q
        for param in self.backbone.parameters():
            param.requires_grad = False

        # self.last_channel = self.backbone.fc.out_features
        self.last_channel = self.backbone.classifier.out_features
        self.classifier = nn.Sequential(
            # nn.Dropout(0.2),
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
        return {
            "val_loss": loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "output": output.detach(),
            "target": y.detach(),
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")
        outputs = torch.stack([x["output"] for x in outputs]).flatten().numpy()
        targets = torch.stack([x["target"] for x in outputs]).flatten().numpy()

        val_f1 = f1(outputs, targets, num_classes=self.num_classes)
        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "val_f1": val_f1,
        }
        self.log_dict(log)

    def configure_optimizers(self):
        # return optim.AdamW(self.parameters(), lr=self.learning_rate)
        optimizer = optim.Adam(self.parameters(), self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer)
        return [optimizer], [scheduler]


class DALISSLCOVIDNet(SSLCOVIDNet):
    def __init__(self,
                 moco_extractor: Moco_v2,
                 data_dir: str,
                 num_workers: Optional[int] = 4,
                 sampling_ratio: Optional[float] = 1.,
                 random_sampling: Optional[bool] = False,
                 num_classes: Optional[int] = 3,
                 batch_size: Optional[int] = 32,
                 lr: Optional[float] = 1e-4):
        super().__init__(moco_extractor, num_classes, batch_size, lr)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.sampling_ratio = sampling_ratio
        self.random_sampling = random_sampling

    def prepare_data(self):
        device_id = self.local_rank
        shard_id = self.global_rank
        num_shards = self.trainer.world_size

        class LightningWrapper(DALIGenericIteratorV2):
            def __init__(self, *kargs, **kvargs):
                super().__init__(*kargs, **kvargs)

            def __next__(self):
                out = super().__next__()
                q = out[0]["img_q"]
                k = out[0]["img_k"]
                label = out[0]["label"].squeeze(-1)
                return (q, k), label

        train_eii = SSLCOVIDxCTIterator(
            self.data_dir, split="train",
            batch_size=self.batch_size,
            sampling_ratio=self.sampling_ratio,
            random_sampling=self.random_sampling)
        val_eii = SSLCOVIDxCTIterator(
            self.data_dir, split="val",
            batch_size=self.batch_size,
            sampling_ratio=1.,
            random_sampling=self.random_sampling)
        test_eii = SSLCOVIDxCTIterator(
            self.data_dir, split="test",
            batch_size=self.batch_size,
            sampling_ratio=1.,
            random_sampling=self.random_sampling)

        train_pipeline = ssl_covidxct_train_pipeline(
            train_eii, height=224,
            device="gpu",
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=self.batch_size,
            num_threads=self.num_workers)
        val_pipeline = ssl_covidxct_val_pipeline(
            val_eii, height=224,
            device="gpu",
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=self.batch_size,
            num_threads=self.num_workers)
        test_pipeline = ssl_covidxct_val_pipeline(
            test_eii, height=224,
            device="gpu",
            device_id=device_id,
            shard_id=shard_id,
            num_shards=num_shards,
            batch_size=self.batch_size,
            num_threads=self.num_workers)

        self.train_loader = LightningWrapper(
            len(train_eii),
            self.batch_size,
            train_pipeline,
            ["img_q", "img_k", "label"],
            auto_reset=True)
        self.val_loader = LightningWrapper(
            len(val_eii),
            self.batch_size,
            val_pipeline,
            ["img_q", "img_k", "label"],
            auto_reset=True)
        self.test_loader = LightningWrapper(
            len(test_eii),
            self.batch_size,
            test_pipeline,
            ["img_q", "img_k", "label"],
            auto_reset=True)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    # def on_epoch_end(self):
    #     self.train_loader.reset()
