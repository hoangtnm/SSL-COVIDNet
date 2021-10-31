from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI

from datasets import SSLCOVIDxCTDataModule
from models import SSLCOVIDNet


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.data_dir", "data.data_dir")
        parser.link_arguments("model.batch_size", "data.batch_size")
        parser.link_arguments("model.num_workers", "data.num_workers")
        parser.add_lightning_class_args(ModelCheckpoint, "best_loss_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "best_auc_checkpoint")
        parser.add_lightning_class_args(ModelCheckpoint, "last_checkpoint")
        parser.set_defaults({
            "best_loss_checkpoint.monitor": "val_loss",
            "best_loss_checkpoint.filename": "best_loss_{epoch}-{val_loss:.2f}-{val_auc_mean:.2f}",
            "best_loss_checkpoint.mode": "min",
            "best_auc_checkpoint.monitor": "val_auc_mean",
            "best_auc_checkpoint.filename": "best_auc_{epoch}-{val_loss:.2f}-{val_auc_mean:.2f}",
            "best_auc_checkpoint.mode": "max",
            "last_checkpoint.filename": "last_{epoch}-{val_loss:.2f}-{val_auc_mean:.2f}",
        })


def main():
    cli = MyLightningCLI(SSLCOVIDNet, SSLCOVIDxCTDataModule,
                         save_config_overwrite=True)


if __name__ == '__main__':
    main()
