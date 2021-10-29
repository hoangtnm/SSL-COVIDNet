from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.cli import LightningCLI

from datasets import SSLCOVIDxCTDataModule
from models import MoCoV2


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.data_dir", "data.data_dir")
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("model.num_workers", "data.num_workers")
        parser.add_lightning_class_args(ModelCheckpoint, "best_loss_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "best_acc_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "last_checkpoint")
        parser.add_lightning_class_args(LearningRateMonitor, "lr_monitor")
        parser.set_defaults({
            "best_loss_checkpoint.monitor": "val_loss",
            "best_loss_checkpoint.filename": "best_loss_{epoch}-{val_loss:.2f}-{val_acc1:.2f}-{val_acc5:.2f}",
            "best_loss_checkpoint.mode": "min",
            "best_acc_checkpoint.monitor": "val_acc1",
            "best_acc_checkpoint.filename": "best_acc_{epoch}-{val_loss:.2f}-{val_acc1:.2f}-{val_acc5:.2f}",
            "best_acc_checkpoint.mode": "max",
            "last_checkpoint.filename": "last_{epoch}-{val_loss:.2f}-{val_acc1:.2f}-{val_acc5:.2f}",
            "lr_monitor.logging_interval": "step",
        })


def main():
    cli = MyLightningCLI(MoCoV2, SSLCOVIDxCTDataModule,
                         save_config_overwrite=True)


if __name__ == '__main__':
    main()
