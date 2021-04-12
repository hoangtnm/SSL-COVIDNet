from typing import Optional

import torch.nn.functional as F
from pl_bolts.metrics import precision_at_k
from pl_bolts.models.self_supervised import Moco_v2

from datasets.covidxct import SSLCOVIDxCTIterator, ssl_covidxct_train_pipeline, \
    ssl_covidxct_val_pipeline
from utils.dali import DALIGenericIteratorV2


class DALIMoco_v2(Moco_v2):
    def __init__(self,
                 data_dir: str,
                 num_workers: Optional[int] = 8,
                 batch_size: Optional[int] = 4,
                 sampling_ratio: Optional[float] = 1.,
                 random_sampling: Optional[bool] = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.sampling_ratio = sampling_ratio
        self.random_sampling = random_sampling

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {'train_loss': loss, 'train_acc1': acc1, 'train_acc5': acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), labels = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        return {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}

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
