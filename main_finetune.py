import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from datasets import SSLCOVIDxCTDataModule
from models import SSLCOVIDNet
from transforms import FineTuneTrainCTTransforms, FineTuneEvalCTTransforms


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SSLCOVIDxCTDataModule.add_argparse_args(parser)
    parser = SSLCOVIDNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = SSLCOVIDNet(**args.__dict__, epochs=args.max_epochs)
    datamodule = SSLCOVIDxCTDataModule.from_argparse_args(args)
    datamodule.train_transforms = FineTuneTrainCTTransforms(
        height=224, out_channels=model.in_channels,
    )
    datamodule.val_transforms = FineTuneEvalCTTransforms(
        height=224, out_channels=model.in_channels,
    )

    checkpointing = ModelCheckpoint(
        monitor="val_auc_mean",
        filename=f"{model.base_encoder}_"
                 "{epoch}-{val_auc_mean:.2f}",
        save_top_k=1,
        mode="max",
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=args.early_stopping_patience,
    #     mode="min"
    # )
    callbacks = [checkpointing]
    plugins = DDPPlugin(
        find_unused_parameters=False,
    ) if args.gpus >= 2 else None
    logger = TensorBoardLogger(save_dir=os.getcwd(),
                               name="lightning_logs_finetune")
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, callbacks=callbacks, plugins=plugins,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.validate(datamodule=datamodule)


if __name__ == '__main__':
    main()
