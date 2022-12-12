from typing import Tuple, Union

import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


class GDALCallback:
    def __init__(self):
        self.progress = 0

    def __call__(self, completed, message, args):
        progress = int(completed * 100)
        if self.progress > progress:
            self.progress = 0
        for tick in range(self.progress, progress + 1):
            if tick == 100:
                print(tick, '- done.')
            elif tick % 10 == 0:
                print(tick, end='')
            elif tick % 2 == 0:
                print('.', end='', flush=True)
            self.progress += 1
        return True


def draw_one_row(*images, size=1024, output=None):
    try:
        size = size[:2] if len(size) >= 2 else size * 2
        size = tuple(map(int, size))
    except:
        size = (int(size), int(size))
    count = len(images)
    figure, axes = plt.subplots(1, count, dpi=72,
                                figsize=(size[0] / 72, size[1] / 72))
    for i in range(count):
        if i:
            axes[i].imshow(images[i], cmap='gray', vmin=0, vmax=1)
        else:
            axes[i].imshow(images[i])
    if output is not None:
        try:
            mkdir(osp.dirname(output), exist_ok=True)
            plt.savefig(output)
        except:
            pass
    plt.show()

def adjust_gamma(image: np.ndarray, gamma: float = 1.1,
                 pad: Union[int, Tuple[int, int]] = 1) -> np.ndarray:
    assert image.ndim == 2, f"Input image must be grayscale!"
    if type(pad) is Tuple[int, int]:
        pad_low, pad_high = pad
    else:
        pad_low, pad_high = (pad, 0)
    lut = ((np.linspace(pad_low, 255 - pad_high, 256) / 255) **
           (1 / gamma) * 255).round().astype(np.uint8)
    return cv.LUT(image, lut).astype(np.uint8)

def apply_kmeans(image: np.ndarray, num_clusters: int, cycles: int = 10,
                 iters: int = 10, eps: float = 0.9,
                 mask: int = 255) -> np.ndarray:
    assert image.ndim == 2, f"Image must be 2D, but {image.ndim}D is given!"
    # Samples are the float32 image with the mask color dropped to zero
    samples = (np.float32(image) *
               (image != mask).astype(np.uint8)).reshape(-1, 1)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iters, eps)
    _, labels, centers = cv.kmeans(samples, num_clusters, None, criteria,
                                   10, cv.KMEANS_PP_CENTERS)
    spread = np.linspace(0, 255, centers.shape[0] + 1)\
                         [centers.argsort(axis=0)].round().astype(np.uint8)
    return spread[labels.flatten()].reshape(image.shape)

def extend_mask(image: np.ndarray, n: int = -5, color: int = 255) -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8)
    if n > 0:
        lut[:n] = color
    elif n < 0:
        lut[n:] = color
    return cv.LUT(image, lut).astype(np.uint8)
