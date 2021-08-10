from argparse import ArgumentParser
from functools import partial
from typing import Union

import monai
import torch.nn as nn
import torch.optim as optim
import torchvision
from pl_bolts.metrics import mean
from pl_bolts.models.self_supervised import Moco_v2


class MoCoV2(Moco_v2):
    def __init__(self,
                 base_encoder: Union[str, nn.Module] = 'resnet18',
                 in_channels: int = 3,
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
                 *args,
                 **kwargs):
        super().__init__(base_encoder,
                         emb_dim,
                         num_negatives,
                         encoder_momentum,
                         softmax_temperature,
                         learning_rate,
                         momentum,
                         weight_decay,
                         data_dir,
                         batch_size,
                         use_mlp,
                         num_workers,
                         in_channels=in_channels,
                         *args,
                         **kwargs)

    def init_encoders(self, base_encoder):
        try:
            template_model = getattr(torchvision.models, base_encoder)
            template_model = partial(
                template_model,
                num_classes=self.hparams.emb_dim,
            )
            assert self.hparams.in_channels == 3, \
                "torchvision models only supports 3 channels"
        except AttributeError:
            template_model = getattr(monai.networks.nets, base_encoder)
            template_model = partial(
                template_model,
                spatial_dims=2,
                in_channels=self.hparams.in_channels,
                out_channels=self.hparams.emb_dim,
            )

        encoder_q = template_model()
        encoder_k = template_model()
        return encoder_q, encoder_k

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {
            "val_loss": val_loss,
            "val_acc1": val_acc1,
            "val_acc5": val_acc5,
            "hp_metric": val_acc1,
        }
        self.log_dict(log)

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.trainer.max_epochs,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_encoder', type=str, default='densenet121')
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--emb_dim', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_negatives', type=int, default=65536)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--data_dir', type=str, default='./')
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--use_mlp', action='store_true')

        return parser
