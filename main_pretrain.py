import argparse
import pytorch_lightning as pl
from pl_bolts.models.self_supervised import MocoV2
from pl_bolts.models.self_supervised.moco.transforms import Moco2TrainImagenetTransforms, Moco2EvalImagenetTransforms
from datasets import SSLCOVIDxCT


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MocoV2.add_model_specific_args(parser)
    args = parser.parse_args()

    datamodule = SSLCOVIDxCT.from_argparse_args(args)
    datamodule.train_transforms = Moco2TrainImagenetTransforms(height=224)
    datamodule.val_transforms = Moco2EvalImagenetTransforms(height=224)
    # datamodule.prepare_data()

    model = MocoV2(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.tune(model, datamodule=datamodule)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
