from pathlib import Path
from typing import Any, Callable, Optional, Iterable, Union

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from nvidia.dali import pipeline_def
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transforms import moco2_train_imagenet_transforms, \
    moco2_val_imagenet_transforms


class UnlabeledCOVIDxCT(Dataset):
    def __init__(self, root: str, split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 sampling_ratio: Optional[float] = 1.,
                 **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "val", "test"]
        self.root = Path(root).expanduser()
        self.df = pd.read_csv(
            self.root / f"{split}_COVIDx_CT-2A.txt", delimiter=" ",
            names=["filename", "class", "xmin", "ymin", "xmax", "ymax"])
        self.df = self.df.sample(
            frac=sampling_ratio, random_state=1000).reset_index(drop=True)
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
                 num_workers: Optional[int] = 4,
                 batch_size: Optional[int] = 32,
                 shuffle: Optional[bool] = False,
                 sampling_ratio: Optional[float] = 1.,
                 random_sampling: Optional[bool] = False,
                 pin_memory: Optional[bool] = True,
                 drop_last: Optional[bool] = True,
                 *args: Any,
                 **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampling_ratio = sampling_ratio
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
            self.covidxct_train = UnlabeledCOVIDxCT(self.data_dir,
                                                    split="train",
                                                    transform=train_transforms,
                                                    sampling_ratio=self.sampling_ratio)
            self.covidxct_val = UnlabeledCOVIDxCT(self.data_dir, split="val",
                                                  transform=val_transforms)

        if stage == "test":
            test_transforms = self._default_transforms() if self.test_transforms is None else self.test_transforms
            self.covidxct_test = UnlabeledCOVIDxCT(self.data_dir, split="test",
                                                   transform=test_transforms)

    def train_dataloader(self) -> DataLoader:
        sampler = WeightedRandomSampler(
            self.covidxct_train.sampling_weights,
            num_samples=len(self.covidxct_train)
        ) if self.random_sampling else None
        return DataLoader(
            self.covidxct_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            sampler=sampler,
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


class SSLCOVIDxCTIterator:
    def __init__(self, root: str, batch_size: int, split: str = "train",
                 sampling_ratio: Optional[float] = 1.,
                 random_sampling: Optional[bool] = False, **kwargs):
        assert split in ["train", "val", "test"]
        self.root = Path(root).expanduser()
        self.batch_size = batch_size
        self.df = pd.read_csv(self.root / f"{split}_COVIDx_CT-2A.txt",
                              delimiter=" ",
                              names=["filename", "class", "xmin", "ymin",
                                     "xmax", "ymax"])
        self.df = self.df.sample(frac=sampling_ratio,
                                 random_state=1000).reset_index(drop=True)
        self.sampling_ratio = sampling_ratio
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
            xmin, ymin = data["xmin"], data["ymin"]
            xmax, ymax = data["xmax"], data["ymax"]
            imgs.append(np.frombuffer(f.read(), dtype=np.uint8))
            bboxes.append(np.array([xmin, ymin, xmax, ymax], dtype=np.uint16))
            labels.append(np.array([data["class"]], dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return imgs, labels, bboxes

    def __len__(self):
        return len(self.df)

    next = __next__


@pipeline_def
def ssl_covidxct_train_pipeline(source: Union[Callable, Iterable],
                                height: Optional[int] = 224,
                                output_layout: Optional[str] = "CHW",
                                device="gpu", shard_id=0, num_shards=1):
    encoded_imgs, labels, bboxes = fn.external_source(source, num_outputs=3)
    shapes = fn.peek_image_shape(encoded_imgs)
    h = fn.slice(shapes, 0, 1, axes=[0])
    w = fn.slice(shapes, 1, 1, axes=[0])
    decoded_imgs = fn.decoders.image(encoded_imgs,
                                     device="mixed" if device == "gpu" else "cpu",
                                     output_type=types.RGB)
    xmin = fn.cast(fn.slice(bboxes, 0, 1, axes=[0]), dtype=types.FLOAT)
    ymin = fn.cast(fn.slice(bboxes, 1, 1, axes=[0]), dtype=types.FLOAT)
    xmax = fn.cast(fn.slice(bboxes, 2, 1, axes=[0]), dtype=types.FLOAT)
    ymax = fn.cast(fn.slice(bboxes, 3, 1, axes=[0]), dtype=types.FLOAT)
    imgs = fn.crop(decoded_imgs,
                   crop_pos_x=xmin / (w - height),
                   crop_pos_y=ymin / (h - height),
                   crop_w=(xmax - xmin),
                   crop_h=(ymax - ymin))
    q = moco2_train_imagenet_transforms(imgs, height, output_layout)
    k = moco2_train_imagenet_transforms(imgs, height, output_layout)
    if device == "gpu":
        labels = labels.gpu()
    labels = fn.cast(labels, dtype=types.INT64)
    return q, k, labels


@pipeline_def
def ssl_covidxct_val_pipeline(source: Union[Callable, Iterable],
                              height: Optional[int] = 224,
                              output_layout: Optional[str] = "CHW",
                              device="gpu", shard_id=0, num_shards=1):
    encoded_imgs, labels, bboxes = fn.external_source(source, num_outputs=3)
    shapes = fn.peek_image_shape(encoded_imgs)
    h = fn.slice(shapes, 0, 1, axes=[0])
    w = fn.slice(shapes, 1, 1, axes=[0])
    decoded_imgs = fn.decoders.image(encoded_imgs,
                                     device="mixed" if device == "gpu" else "cpu",
                                     output_type=types.RGB)
    xmin = fn.cast(fn.slice(bboxes, 0, 1, axes=[0]), dtype=types.FLOAT)
    ymin = fn.cast(fn.slice(bboxes, 1, 1, axes=[0]), dtype=types.FLOAT)
    xmax = fn.cast(fn.slice(bboxes, 2, 1, axes=[0]), dtype=types.FLOAT)
    ymax = fn.cast(fn.slice(bboxes, 3, 1, axes=[0]), dtype=types.FLOAT)
    imgs = fn.crop(decoded_imgs,
                   crop_pos_x=xmin / (w - height),
                   crop_pos_y=ymin / (h - height),
                   crop_w=(xmax - xmin),
                   crop_h=(ymax - ymin))
    q = moco2_val_imagenet_transforms(imgs, height, output_layout)
    k = moco2_val_imagenet_transforms(imgs, height, output_layout)
    if device == "gpu":
        labels = labels.gpu()
    labels = fn.cast(labels, dtype=types.INT64)
    return q, k, labels
