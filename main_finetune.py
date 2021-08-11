import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from datasets import SSLCOVIDxCTDataModule
from models import SSLCOVIDNet
from transforms import FineTuneTrainCTTransforms, FineTuneEvalCTTransforms


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    parser.add_argument("--random_sampling", action="store_true",
                        help="Whether to user Random Sampling when fine-tuning")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SSLCOVIDNet.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = SSLCOVIDxCTDataModule.from_argparse_args(args)
    datamodule.train_transforms = FineTuneTrainCTTransforms(
        height=224, out_channels=args.in_channels
    )
    datamodule.val_transforms = FineTuneEvalCTTransforms(
        height=224, out_channels=args.in_channels,
    )

    model = SSLCOVIDNet(
        **args.__dict__,
        num_classes=len(datamodule.pathology_list),
        val_pathology_list=datamodule.pathology_list,
        pretrained_checkpoint=args.pretrained_checkpoint,
        epochs=args.max_epochs,
    )
    checkpointing = ModelCheckpoint(
        monitor="val_auc_mean",
        filename="{epoch}-{val_auc_mean:.2f}",
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
        find_unused_parameters=False
    ) if args.gpus >= 2 else None
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, plugins=plugins
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
