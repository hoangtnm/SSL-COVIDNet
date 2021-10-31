import random
from typing import Tuple

import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torch import Tensor


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class ToRGB:
    """Converts the input grayscale image to RGB."""

    def __call__(self, img):
        return Image.fromarray(img).convert("RGB")


class MocoTrainTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = T.Compose([
            ToRGB(),
            T.RandomResizedCrop(target_size, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([0.1, 2.0])]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class MocoEvalTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = T.Compose([
            ToRGB(),
            T.Resize(target_size + 32),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
        ])

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class FinetuneTrainTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = T.Compose([
            ToRGB(),
            T.RandomResizedCrop(target_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
        ])

    def __call__(self, inp: np.ndarray) -> Tensor:
        return self.transform(inp)


class FinetuneEvalTransforms:
    def __init__(self, target_size: int = 224):
        self.transform = T.Compose([
            ToRGB(),
            T.Resize(target_size + 32),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize((0.673, 0.673, 0.673), (0.327, 0.327, 0.327)),
        ])

    def __call__(self, inp: np.ndarray) -> Tensor:
        return self.transform(inp)
