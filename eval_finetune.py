from argparse import ArgumentParser

import pytorch_lightning as pl
import yaml

from datasets import SSLCOVIDxCTDataModule
from models import SSLCOVIDNet


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrained", required=True)
    args = parser.parse_args()

    config = yaml.load(args.config, yaml.FullLoader)
    assert config["data"]["phase"] == "finetune"
    model = SSLCOVIDNet.load_from_checkpoint(args.pretrained)
    datamodule = SSLCOVIDxCTDataModule(**config["data"])
    trainer = pl.Trainer(**config["trainer"])
    trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
