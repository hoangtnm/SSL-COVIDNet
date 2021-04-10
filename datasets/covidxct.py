from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from nvidia.dali import pipeline_def
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils import mux


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
        data = self.df.iloc[idx]
        filepath = Path(self.root) / "2A_images" / data["filename"]
        xmin, ymin, xmax, ymax = data["xmin"], data["ymin"], data["xmax"], data["ymax"]
        label = data["class"]

        img = Image.open(filepath).convert("RGB")
        img = img.crop((xmin, ymin, xmax, ymax))
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


class SSLCOVIDxCT(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str,
                 num_workers: Optional[int] = 8,
                 batch_size: Optional[int] = 32,
                 shuffle: Optional[bool] = False,
                 random_sampling: Optional[bool] = False,
                 pin_memory: Optional[bool] = False,
                 drop_last: Optional[bool] = True,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_sampling = random_sampling
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = self._default_transforms() if self.train_transforms is None else self.train_transforms
            val_transforms = self._default_transforms() if self.val_transforms is None else self.val_transforms
            self.covidxct_train = UnlabeledCOVIDxCT(self.data_dir, split="train", transform=train_transforms)
            self.covidxct_val = UnlabeledCOVIDxCT(self.data_dir, split="val", transform=val_transforms)

        if stage == "test":
            test_transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms
            self.covidxct_test = UnlabeledCOVIDxCT(self.data_dir, split="test", transform=test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.covidxct_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=WeightedRandomSampler(self.covidxct_train.sampling_weights,
                                          num_samples=(len(self.covidxct_train))) if self.random_sampling else None,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.covidxct_val,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.covidxct_test,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )

    def _default_transforms(self) -> Callable:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )


class UnlabeledCOVIDxCTIterator:
    def __init__(self, root: str, batch_size: int, split: str = "train", random_sampling: Optional[bool] = False,
                 **kwargs):
        assert split in ["train", "val", "test"]
        self.root = Path(root).expanduser()
        self.batch_size = batch_size
        self.df = pd.read_csv(self.root / f"{split}_COVIDx_CT-2A.txt", delimiter=" ",
                              names=["filename", "class", "xmin", "ymin", "xmax", "ymax"])
        self.df = self.df.sample(frac=1, random_state=1000).reset_index(drop=True)
        self.random_sampling = random_sampling

    @property
    def sampling_weights(self) -> np.ndarray:
        labels = self.df.loc[:, "class"]
        counts = labels.value_counts()
        weights = 1. / counts
        return weights[labels].to_numpy()

    def __iter__(self):
        self.i = 0
        self.n = len(self.df)

        if self.random_sampling:
            weights = self.sampling_weights
            sampler = WeightedRandomSampler(weights, num_samples=self.n)
            self.sampling_indices = list(sampler)
            return self

        self.sampling_indices = list(range(self.n))
        return self

    def __next__(self):
        imgs = []
        labels = []
        bboxes = []

        if self.i >= self.n:
            self.__iter__()
            raise StopIteration

        for _ in range(self.batch_size):
            sampling_idx = self.sampling_indices[self.i % self.n]
            data = self.df.iloc[sampling_idx]
            filepath = Path(self.root) / "2A_images" / data["filename"]
            f = open(filepath, "rb")
            xmin, ymin, xmax, ymax = data["xmin"], data["ymin"], data["xmax"], data["ymax"]
            imgs.append(np.frombuffer(f.read(), dtype=np.uint8))
            bboxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.uint8))
            labels.append(np.array([data["class"]], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return imgs, labels, bboxes


def random_grayscale(imgs, probability):
    saturate = fn.random.coin_flip(probability=1 - probability)
    saturate = fn.cast(saturate, dtype=types.DALIDataType.FLOAT)
    return fn.hsv(imgs, saturation=saturate)


@pipeline_def
def hybrid_covidxct_train_pipe(height: Optional[int] = 224):
    imgs, labels, bboxes = fn.external_source(source=None, num_outputs=3)
    xmin, ymin, xmax, ymax = bboxes
    imgs = fn.decoders.image(imgs, device="mixed", output_type=types.RGB)
    jitter_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
    blur_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.5)
    flip_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.5)

    imgs = fn.crop(imgs, crop_pos_x=xmin, crop_pos_y=ymin, crop_w=(xmax - xmin), crop_h=(ymax - ymin))
    imgs = fn.random_resized_crop(imgs, size=height, random_area=(.2, 1.))
    jittered_imgs = fn.color_twist(imgs, brightness=0.4, constrast=0.4, saturation=0.4, hue=0.1)

    imgs = mux(jitter_condition, jittered_imgs, imgs)
    imgs = random_grayscale(imgs, probability=0.2)

    blurred_imgs = fn.gaussian_blur(imgs, sigma=(.1, 2.))
    imgs = mux(blur_condition, blurred_imgs, imgs)

    # flipped_imgs = fn.flip(imgs)
    # imgs = mux(flip_condition, flipped_imgs, imgs)
    imgs = imgs / 255.
    imgs = fn.crop_mirror_normalize(imgs,
                                    dtype=types.FLOAT,
                                    mirror=flip_condition,
                                    output_layout="CHW",
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return imgs, labels
