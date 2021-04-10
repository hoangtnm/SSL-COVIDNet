import argparse
from pathlib import Path

import pytorch_lightning as pl
from pl_bolts.models.self_supervised import Moco_v2
from pl_bolts.models.self_supervised.moco.transforms import Moco2TrainImagenetTransforms, Moco2EvalImagenetTransforms
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
# from pytorch_lightning.loggers import TensorBoardLogger

from datasets import SSLCOVIDxCT
from models import SSLCOVIDNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="pretrain", choices=["pretrain", "finetuning"], help="")
    parser.add_argument("--base_encoder_checkpoint", help="")
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--random_sampling", action="store_true")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Moco_v2.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = SSLCOVIDxCT.from_argparse_args(args)
    datamodule.train_transforms = Moco2TrainImagenetTransforms(height=224)
    datamodule.val_transforms = Moco2EvalImagenetTransforms(height=224)
    # datamodule.prepare_data()

    if args.phase == "pretrain":
        model = Moco_v2(**args.__dict__)
    else:
        assert Path(args.base_encoder_checkpoint).exists()
        feature_extractor = Moco_v2(**args.__dict__)
        feature_extractor = feature_extractor.load_from_checkpoint(args.base_encoder_checkpoint)
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
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, early_stop_callback])
    # trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
