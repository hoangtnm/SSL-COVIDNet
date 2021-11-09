from typing import Tuple

import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
from albumentations import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2
from torch import Tensor

from .simclr import GaussianBlur


class AddChannel(ImageOnlyTransform):
    """Adds a 1-length channel dimension to the input image."""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return img[..., None]


class RepeatChannel(ImageOnlyTransform):
    """Repeat channel data to construct expected input shape for models."""

    def __init__(self, repeats: int, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        if repeats <= 0:
            raise AssertionError("repeats count must be greater than 0.")
        self.repeats = repeats

    def apply(self, img, **params):
        return np.repeat(img, self.repeats, -1)


class ToRGB(ImageOnlyTransform):
    """Converts the input grayscale image to RGB."""

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        img = Image.fromarray(img).convert("RGB")
        return np.array(img)


class ToTensor(ImageOnlyTransform):
    """Convert a PIL Image or numpy.ndarray to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    """

    def __init__(self, always_apply=False, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return F.to_tensor(img)


class MocoTrainCTTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            AddChannel(),
            A.RandomResizedCrop(target_size, target_size, scale=(0.2, 1.0)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            # A.Rotate(limit=15),
            # A.CLAHE(clip_limit=2),
            # A.GaussianBlur(),
            # A.GaussNoise((4.0, 8.0)),
            GaussianBlur(),
            RepeatChannel(3),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(image=inp)["image"]
        k = self.transform(image=inp)["image"]
        return q, k


class MocoEvalCTTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            AddChannel(),
            A.Resize(target_size + 32, target_size + 32),
            A.CenterCrop(target_size, target_size),
            # A.CLAHE(clip_limit=2),
            RepeatChannel(3),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(image=inp)["image"]
        k = self.transform(image=inp)["image"]
        return q, k


class MocoTrainTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            ToRGB(),
            A.RandomResizedCrop(target_size, target_size, scale=(0.2, 1.0)),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.ToGray(p=0.2),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            GaussianBlur(),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(image=inp)["image"]
        k = self.transform(image=inp)["image"]
        return q, k


class MocoEvalTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            ToRGB(),
            A.Resize(target_size + 32, target_size + 32),
            A.CenterCrop(target_size, target_size),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(image=inp)["image"]
        k = self.transform(image=inp)["image"]
        return q, k


class FinetuneTrainTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            ToRGB(),
            A.RandomResizedCrop(target_size, target_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tensor:
        return self.transform(image=inp)["image"]


class FinetuneEvalTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = A.Compose([
            ToRGB(),
            A.Resize(target_size + 32, target_size + 32),
            A.CenterCrop(target_size, target_size),
            A.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
            ToTensorV2(),
        ])

    def __call__(self, inp: np.ndarray) -> Tensor:
        return self.transform(image=inp)["image"]
