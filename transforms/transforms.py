from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch import Tensor


class BodyCrop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, label, coords):
        xmin, ymin, xmax, ymax = coords
        img = F.crop(img, ymin, xmin, (ymax - ymin), (xmax - xmin))
        return img, label


class Moco2TrainCovidxCT(nn.Module):
    def __init__(self, height: int = 128):
        super().__init__()
        self.T = nn.Sequential(
            BodyCrop(),
            T.RandomSizedCrop(height, scale=(0.2, 1.)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=2, sigma=(.1, 2.))], p=0.5),
            T.RandomHorizontalFlip(),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class Moco2EvalCovidxCT(nn.Module):
    def __init__(self, height: int = 128):
        super().__init__()
        self.transform = nn.Sequential(
            T.Resize(height + 32),
            T.CenterCrop(height),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def forward(self, inp: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k
