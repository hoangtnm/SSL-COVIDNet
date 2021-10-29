import random

import numpy as np
from PIL import Image, ImageFilter
from albumentations import ImageOnlyTransform


class GaussianBlur(ImageOnlyTransform):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=(0.1, 2.0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.sigma = sigma

    def apply(self, img, **params):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        num_channels = img.shape[-1]
        if num_channels == 1:
            img = np.squeeze(img, axis=-1)

        img = Image.fromarray(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = np.array(img)

        if num_channels == 1:
            img = img[..., None]
        return img
