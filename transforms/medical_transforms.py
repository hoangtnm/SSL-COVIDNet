from typing import Tuple

import monai.transforms as T
import numpy as np
from scipy.ndimage import gaussian_filter
from torch import Tensor


# import torchio as tio
# from monai.transforms.adaptors import adaptor


# class AddDepthChannel:
#     def __call__(self, img: np.ndarray):
#         return img[..., None]


class RandomGaussianBlur:
    """Random Gaussian blur transform.

    Args:
        p: Probability to apply transform.
        sigma_range: Range of sigma values for Gaussian kernel.
    """

    def __init__(self, p: float = 0.5,
                 sigma_range: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.uniform() <= self.p:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
            img = gaussian_filter(img, sigma).astype(img.dtype)
        return img


class AddGaussianNoise:
    """Gaussian noise transform.

    Args:
        p: Probability of adding Gaussian noise.
        snr_range: SNR range for Gaussian noise addition.
    """

    def __init__(self, p: float = 0.5,
                 snr_range: Tuple[float, float] = (2.0, 8.0)):
        self.p = p
        self.snr_range = snr_range

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.uniform() <= self.p:
            snr_level = np.random.uniform(low=self.snr_range[0],
                                          high=self.snr_range[1])
            signal_level = np.mean(img)

            # use numpy to keep things consistent on numpy random seed
            img = img \
                  + (signal_level / snr_level) \
                  * np.random.normal(size=tuple(img.shape)).astype(img.dtype)
        return img


class HistogramNormalize:
    """Applies histogram normalization.

    Args:
        bins: Number of histogram bins.

    Returns:
        Histogram represented as a tensor.
    """

    def __init__(self, bins: int = 256):
        self.number_bins = bins

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_histogram, bins = np.histogram(
            img.flatten(), self.number_bins, density=True
        )
        cdf = img_histogram.cumsum()  # cumulative distribution
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        img_equalized = np.interp(img.flatten(), bins[:-1], cdf)
        return img_equalized.reshape(img.shape).astype(img.dtype)


class MocoTrainCTTransforms:
    def __init__(self, height: int = 224, out_channels: int = 3):
        transforms = [
            T.AddChannel(),
            T.ToTensor(),
            T.TorchVision("RandomResizedCrop", height, scale=(0.2, 1.0)),
            T.ToNumpy(),
            T.RandFlip(0.5, spatial_axis=(0, 1)),
            T.CastToType(np.float32),
            T.ScaleIntensityRange(0, 255, 0, 1),
            RandomGaussianBlur(),
            T.RandGaussianNoise(0.5),
            # HistogramNormalize(),
            T.ToTensor(),
        ]
        if out_channels == 3:
            transforms.insert(-1, T.RepeatChannel(3))
        self.transform = T.Compose(transforms)

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k


class MocoEvalCTTransforms:
    def __init__(self, height: int = 224, out_channels: int = 3):
        transforms = [
            T.AddChannel(),
            T.Resize((height + 32, height + 32)),
            T.CenterSpatialCrop((224, 224)),
            T.CastToType(np.float32),
            T.ScaleIntensityRange(0, 255, 0, 1),
            # HistogramNormalize(),
            T.ToTensor(),
        ]
        if out_channels == 3:
            transforms.insert(-1, T.RepeatChannel(3))
        self.transform = T.Compose(transforms)

    def __call__(self, inp: np.ndarray) -> Tuple[Tensor, Tensor]:
        q = self.transform(inp)
        k = self.transform(inp)
        return q, k
