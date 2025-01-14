from argparse import ArgumentParser
from functools import partial
from typing import Union

import monai
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from pl_bolts.metrics import mean, precision_at_k
from pl_bolts.utils import _PL_GREATER_EQUAL_1_4


class MoCoV2(pl.LightningModule):
    def __init__(self,
                 base_encoder: Union[str, nn.Module] = "densenet121",
                 emb_dim: int = 256,
                 num_negatives: int = 65536,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 data_dir: str = './',
                 batch_size: int = 256,
                 use_mlp: bool = False,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k = self.init_encoders(base_encoder)

        if use_mlp:  # hack: brute-force replacement
            if hasattr(self.encoder_q, "fc"):  # ResNet models
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
                )
                self.encoder_k.fc = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
                )
            elif hasattr(self.encoder_q, "classifier"):  # Densenet models
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                    self.encoder_q.classifier
                )
                self.encoder_k.classifier = nn.Sequential(
                    nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),
                    self.encoder_k.classifier
                )

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the validation queue
        self.register_buffer("val_queue", torch.randn(emb_dim, num_negatives))
        self.val_queue = nn.functional.normalize(self.val_queue, dim=0)

        self.register_buffer("val_queue_ptr", torch.zeros(1, dtype=torch.long))

    def init_encoders(self, base_encoder):
        try:
            template_model = getattr(torchvision.models, base_encoder)
            template_model = partial(
                template_model, num_classes=self.hparams.emb_dim,
            )
        except AttributeError:
            template_model = getattr(monai.networks.nets, base_encoder)
            template_model = partial(
                template_model, spatial_dims=2, in_channels=3,
                out_channels=self.hparams.emb_dim,
            )

        encoder_q = template_model()
        encoder_k = template_model()
        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1.0 - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue_ptr, queue):
        # gather keys before updating queue
        if self._use_ddp_or_ddp2(self.trainer):
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):  # pragma: no cover
        """Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):  # pragma: no cover
        """Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, img_q, img_k, queue):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
            queue: a queue from which to pick negative samples
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys

            # shuffle for making use of BN
            if self._use_ddp_or_ddp2(self.trainer):
                img_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(img_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            if self._use_ddp_or_ddp2(self.trainer):
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return logits, labels, k

    def training_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        self._momentum_update_key_encoder()  # update the key encoder
        output, target, keys = self(img_q=img_1, img_k=img_2, queue=self.queue)
        self._dequeue_and_enqueue(keys, queue=self.queue,
                                  queue_ptr=self.queue_ptr)

        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        (img_1, img_2), _ = batch

        output, target, keys = self(img_q=img_1, img_k=img_2,
                                    queue=self.val_queue)
        self._dequeue_and_enqueue(keys, queue=self.val_queue,
                                  queue_ptr=self.val_queue_ptr)

        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)
        self.log("hp_metric", val_acc1)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def _use_ddp_or_ddp2(trainer: pl.Trainer) -> bool:
        # for backwards compatibility
        if _PL_GREATER_EQUAL_1_4:
            return trainer.accelerator_connector.use_ddp or trainer.accelerator_connector.use_ddp2
        return trainer.use_ddp or trainer.use_ddp2


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in
                      range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    return torch.cat(tensors_gather, dim=0)
