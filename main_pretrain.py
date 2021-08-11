import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from datasets import SSLCOVIDxCTDataModule
from models import MoCoV2
from transforms import MocoTrainCTTransforms, MocoEvalCTTransforms


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    parser.add_argument("--random_sampling", action="store_true",
                        help="Whether to user Random Sampling when fine-tuning")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MoCoV2.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = SSLCOVIDxCTDataModule.from_argparse_args(args)
    datamodule.train_transforms = MocoTrainCTTransforms(
        height=224, out_channels=args.in_channels
    )
    datamodule.val_transforms = MocoEvalCTTransforms(
        height=224, out_channels=args.in_channels,
    )

    model = MoCoV2(**args.__dict__)
    best_checkpointing = ModelCheckpoint(
        monitor="val_acc1",
        filename=f"best_{args.base_encoder}" \
                 "{epoch}-{val_loss:.2f}-{val_acc1:.2f}",
        save_top_k=1,
        mode="max",
    )
    checkpointing = ModelCheckpoint(
        monitor="val_acc1",
        filename=f"{args.base_encoder}" \
                 "{epoch}-{val_loss:.2f}-{val_acc1:.2f}",
        save_top_k=-1,
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=args.early_stopping_patience,
    #     mode="min"
    # )
    callbacks = [best_checkpointing, checkpointing]
    plugins = DDPPlugin(
        find_unused_parameters=False
    ) if args.gpus >= 2 else None
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=callbacks, plugins=plugins
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
