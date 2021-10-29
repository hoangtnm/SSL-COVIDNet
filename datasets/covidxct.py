from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transforms.albumentations import MocoTrainTransforms, MocoEvalTransforms


# from transforms.albumentations import MocoTrainCTTransforms, \
#     MocoEvalCTTransforms
# from transforms.monai import MocoTrainCTTransforms, MocoEvalCTTransforms


class UnlabeledCOVIDxCT(Dataset):
    def __init__(self,
                 root: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 sampling_ratio: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        assert split in {"train", "val", "test"}
        self.root = Path(root).expanduser()
        self.split = split
        self.sampling_ratio = sampling_ratio if split == "train" else 1.0

        df = pd.read_csv(
            self.root / f"{split}_COVIDx_CT-2A.txt", delimiter=" ",
            names=["filename", "class", "xmin", "ymin", "xmax", "ymax"]
        )
        self.df = df.sample(
            frac=self.sampling_ratio, random_state=1000
        ).reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        filepath = Path(self.root) / "2A_images" / data["filename"]
        xmin, ymin = data["xmin"], data["ymin"]
        xmax, ymax = data["xmax"], data["ymax"]
        label = data["class"]

        img = Image.open(filepath)
        img = img.crop((xmin, ymin, xmax, ymax))
        img = np.array(img)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label

    @property
    def sampling_weights(self) -> np.ndarray:
        labels = self.df.loc[:, "class"]
        counts = labels.value_counts()
        weights = 1. / counts
        return weights[labels].to_numpy()


class SSLCOVIDxCTDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 num_workers: int = 4,
                 batch_size: int = 32,
                 sampling_ratio: float = 1.0,
                 random_sampling: bool = False,
                 pin_memory: bool = True,
                 drop_last: bool = True,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.sampling_ratio = sampling_ratio
        self.random_sampling = random_sampling
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = self._moco_train_transforms()
            val_transforms = self._moco_val_transforms()
            self.covidxct_train = UnlabeledCOVIDxCT(
                self.data_dir, split="train", transform=train_transforms,
                sampling_ratio=self.sampling_ratio,
            )
            self.covidxct_val = UnlabeledCOVIDxCT(
                self.data_dir, split="val", transform=val_transforms
            )

        if stage == "test":
            test_transforms = self._default_transforms() \
                if self.test_transforms is None else self.test_transforms
            self.covidxct_test = UnlabeledCOVIDxCT(
                self.data_dir, split="test", transform=test_transforms
            )

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(
            self.covidxct_train.sampling_weights,
            num_samples=len(self.covidxct_train)
        ) if self.random_sampling else None

        return DataLoader(
            self.covidxct_train,
            batch_size=self.batch_size,
            shuffle=True,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.covidxct_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.covidxct_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def _moco_train_transforms(self) -> Callable:
        return MocoTrainTransforms(224)

    def _moco_val_transforms(self) -> Callable:
        return MocoEvalTransforms(224)

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        parser = parent_parser.add_argument_group(title="SSLCOVIDxCTDataModule")
        parser.add_argument("--sampling_ratio", type=float, default=1.0)
        parser.add_argument("--random_sampling", action="store_true",
                            help="Whether to use weighted random sampling.")
        return parent_parser
