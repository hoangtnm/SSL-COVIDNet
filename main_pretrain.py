import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pl_bolts.models.self_supervised import Moco_v2
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import SSLCOVIDxCT
from models import SSLCOVIDNet
from transforms import Moco2TrainCovidxCTTransforms, Moco2EvalCovidxCTTransforms

# from pytorch_lightning.loggers import TensorBoardLogger
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="pretrain",
                        choices=["pretrain", "finetune"], help="")
    parser.add_argument("--base_encoder_checkpoint", help="")
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--random_sampling", action="store_true",
                        help="Whether to user Random Sampling when fine-tuning")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Moco_v2.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.phase == "pretrain":
        model = Moco_v2(**args.__dict__)
    else:
        assert Path(args.base_encoder_checkpoint).exists()
        feature_extractor = Moco_v2(**args.__dict__)
        feature_extractor = feature_extractor.load_from_checkpoint(
            args.base_encoder_checkpoint)
        model = SSLCOVIDNet(feature_extractor, num_classes=3)
    # logger = TensorBoardLogger(save_dir=Path(".").absolute(),
    #                            name='lightning_logs',
    #                            default_hp_metric=False)
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=args.early_stopping_patience,
                                        mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          filename=f"{args.base_encoder}" + "{epoch}-{val_loss:.2f}",
                                          save_top_k=1,
                                          mode="min")
    callbacks = [checkpoint_callback, early_stop_callback]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    datamodule = SSLCOVIDxCT.from_argparse_args(args)
    datamodule.train_transforms = Moco2TrainCovidxCTTransforms(height=224)
    datamodule.val_transforms = Moco2EvalCovidxCTTransforms(height=224)
    # datamodule.setup()
    # datamodule.prepare_data()
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)
    # print(datamodule.num_workers)


if __name__ == '__main__':
    main()
