from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class UnlabeledCOVIDxCT(Dataset):
    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val", "test"]
        self.root = Path(root).expanduser()
        self.df = pd.read_csv(self.root / f"{split}_COVIDx_CT-2A.txt", delimiter=" ",
                              names=["filename", "class", "xmin", "ymin", "xmax", "ymax"])
        self.df = self.df.sample(frac=1, random_state=1000).reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filepath = Path(self.root) / "2A_images" / self.df.loc[idx, "filename"]
        label = self.df.loc[idx, "label"]
        img = Image.open(filepath).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label


class SSLCOVIDxCT(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 num_workers: int = 8,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transform = None
        self.val_transform = None
        self.test_transform = None

    @property
    def num_classes(self) -> int:
        return 3

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        transforms = self._default_transform() if self.train_transform is None else self.train_transform
        dataset = UnlabeledCOVIDxCT(
            self.data_dir,
            split="train",
            transform=transforms
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        transforms = self._default_transform() if self.val_transform is None else self.val_transform
        dataset = UnlabeledCOVIDxCT(
            self.data_dir,
            split="val",
            transform=transforms
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        transforms = self._default_transform() if self.test_transform is None else self.test_transform
        dataset = UnlabeledCOVIDxCT(
            self.data_dir,
            split="val",
            transform=transforms
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def _default_transform(self) -> Callable:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
