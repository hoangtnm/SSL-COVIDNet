import random
from typing import Optional
from typing import Tuple

import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import ImageFilter
from torch import Tensor

from utils import mux


class BodyCrop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, coords):
        xmin, ymin, xmax, ymax = coords
        img = F.crop(img, ymin, xmin, (ymax - ymin), (xmax - xmin))
        return img


class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Moco2TrainCovidxCTTransforms(nn.Module):
    def __init__(self, height: int = 128):
        super().__init__()
        self.transform = T.Compose([
            # BodyCrop(),
            T.RandomSizedCrop(height, scale=(0.2, 1.)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur(sigma=(.1, 2.))],
                          p=0.5),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            # T.ConvertImageDtype(torch.float),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class Moco2EvalCovidxCTTransforms(nn.Module):
    def __init__(self, height: int = 128):
        super().__init__()
        self.transform = T.Compose([
            T.Resize(height + 32),
            T.CenterCrop(height),
            # T.ConvertImageDtype(torch.float),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


def random_grayscale(imgs, p: Optional[float] = 0.5):
    # To achieve the grayscale conversion,
    # we will desaturate the input (saturation=0).
    # Therefore, saturation prob will be `1 - p`.
    saturate = fn.random.coin_flip(probability=1 - p)
    saturate = fn.cast(saturate, dtype=types.DALIDataType.FLOAT)
    return fn.hsv(imgs, saturation=saturate)


# def crop_body(decoded_imgs, bboxes, w, h):
#     xmin = fn.cast(fn.slice(bboxes, 0, 1, axes=[0]), dtype=types.FLOAT)
#     ymin = fn.cast(fn.slice(bboxes, 1, 1, axes=[0]), dtype=types.FLOAT)
#     xmax = fn.cast(fn.slice(bboxes, 2, 1, axes=[0]), dtype=types.FLOAT)
#     ymax = fn.cast(fn.slice(bboxes, 3, 1, axes=[0]), dtype=types.FLOAT)
#     return fn.crop(decoded_imgs,
#                    crop_pos_x=xmin / w,
#                    crop_pos_y=ymin / h,
#                    crop_w=(xmax - xmin),
#                    crop_h=(ymax - ymin))


def moco2_train_imagenet_transforms(imgs, height: int,
                                    output_layout: Optional[str] = "CHW"):
    jitter_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
    blur_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.5)
    flip_condition = fn.random.coin_flip(dtype=types.BOOL, probability=0.5)

    imgs = fn.random_resized_crop(imgs, size=height, random_area=(.2, 1.))

    jittered_imgs = fn.color_twist(imgs, brightness=0.4, contrast=0.4,
                                   saturation=0.4, hue=0.1)
    imgs = mux(jitter_condition, jittered_imgs, imgs)

    imgs = random_grayscale(imgs, p=0.2)

    blurred_imgs = fn.gaussian_blur(imgs, sigma=(.1, 2.))
    imgs = mux(blur_condition, blurred_imgs, imgs)

    flipped_imgs = fn.flip(imgs)
    imgs = mux(flip_condition, flipped_imgs, imgs)
    imgs = imgs / 255.
    imgs = fn.crop_mirror_normalize(imgs,
                                    output_layout=output_layout,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return imgs


def moco2_val_imagenet_transforms(imgs, height: int,
                                  output_layout: Optional[str] = "CHW"):
    imgs = fn.resize(imgs, size=height + 32)
    imgs = fn.crop(imgs, crop=[height, height])
    imgs = imgs / 255.
    imgs = fn.crop_mirror_normalize(imgs,
                                    output_layout=output_layout,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return imgs
