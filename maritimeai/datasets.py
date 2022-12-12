from collections import OrderedDict
from functools import reduce
from glob import glob

import cv2 as cv
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose, Lambda, RandomResizedCrop, Resize, ToTensor
)
from torchvision.transforms import InterpolationMode


class DatasetBase(Dataset):
    mode_interpolation = InterpolationMode.NEAREST

    def __init__(self, paths_images, paths_masks, classes, items=None,
                 transformations=None, augmentations=None, size=None,
                 expand=False, flat=False, ext_images='jpg', ext_masks='png',
                 invert=True, empty=False):
        # Assert images paths
        if isinstance(paths_images, str):
            paths_images = (paths_images, path_images)
        elif not isinstance(paths_images, (tuple, list, set)):
            raise TypeError("first argument must be of type str or list!")
        else:
            paths_images = tuple(paths_images)
        # Assert masks paths
        if empty:
            paths_masks = tuple()
        elif isinstance(paths_masks, str):
            paths_masks = (paths_masks,)
        elif not isinstance(paths_masks, (tuple, list, set)):
            raise TypeError("second argument must be of type str or list!")
        else:
            paths_masks = tuple(paths_masks)
        # Default output image/mask size
        if size is None:
            size = (1024, 1024)
        # Default transformations (geometric - image + mask)
        if not isinstance(transformations, (Compose, torch.nn.Module)):
            self.transformations = Compose([
                Resize(size, self.mode_interpolation),
            ])
        else:
            self.transformations = transformations
        # Default augmentations (color - image only)
        if not isinstance(augmentations, (Compose, torch.nn.Module)):
            self.augmentations = Compose([
                Lambda(lambda x: x),
            ])
        else:
            self.augmentations = augmentations
        # Class attributes
        self.to_tensor = ToTensor()
        self.paths_images = paths_images
        self.paths_masks = paths_masks
        if items is None:
            items = []
            for path in paths_images[:2] + paths_masks[:1]:
                # 3 sets
                items.append({osp.splitext(osp.basename(item))[0] for item \
                            in glob(osp.join(path, '*.*'))})
            self.items = sorted(items[-1].intersection(*items))
        else:
            self.items = items
        self.classes = len(classes)
        self.items_class = OrderedDict({c: i for c, i \
                                        in zip(classes,
                                               range(1, self.classes + 1))})
        self.expand = expand
        self.flat = flat
        self.ext_images = ext_images
        self.ext_masks = ext_masks
        self.invert = invert
        # self.empty = empty
        return None

    def __getitem__(self, item):
        image = cv.imread(osp.join(self.paths_images[0],
                                   f"{self.items[item]}.{self.ext_images}"),
                          cv.IMREAD_COLOR)  # -> (h, w, c)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        if self.paths_masks:
            mask = cv.imread(osp.join(self.paths_masks[0],
                                      f"{self.items[item]}.{self.ext_masks}"),
                            cv.IMREAD_GRAYSCALE)  # -> (h, w)
        else:
            mask = np.zeros_like(image[:, :, 0])
        h, w = tuple(map(min, zip(mask.shape[:2], (h, w))))
        # Use minimum height and width 'cause image/mask dimension may not
        # always fit (difference during mask conversion)
        image = Image.fromarray(np.dstack(
            [image[:h, :w, :], np.clip(mask[:h, :w, None], 0, 1)]
        ))
        image = self.transformations(image)
        image = np.array(image)
        image, mask = image[..., :-1], image[..., -1]
        # image = self.to_tensor(np.dstack((image[..., 1], image)))
        image = self.to_tensor(image)
        image = self.augmentations(image)
        if self.invert:
            # print(f"Original mean = {mask.mean()}")  # debug
            mask = 1 - mask  # TODO: scale to mask.max()
            # print(f"Inverted mean = {mask.mean()}")  # debug
        # Convert to int64 'cause OHE requires index tensor
        if self.flat:
            mask = torch.tensor(mask)
            # mask = self.to_tensor(mask)
        elif self.expand:
            mask = torch.nn.functional.one_hot(torch.tensor(mask,
                                                            dtype=torch.int64),
                                               self.classes).to(torch.int8)\
                                               .permute(2, 0, 1)
        else:
            mask = torch.tensor(mask).unsqueeze_(0).to(torch.int64)
        return image, mask

    def __len__(self):
        return len(self.items)

    def debug(self):
        # Debug function to show some dataset stats
        print(f"DEBUG: paths images = {self.paths_images}")
        print(f"DEBUG: paths masks = {self.paths_masks}")
        print(f"DEBUG: items ({len(self.items)}):")
        print('\n'.join(self.items))
        print(f"DEBUG: classes ({self.classes}):")
        print(self.items_class)


class DatasetSamplingProportional(Dataset):
    def __init__(self, *datasets, parts=None, eps=0.1):
        count_datasets = len(datasets)
        if isinstance(parts, (dict, list, tuple)):
            count_parts = len(parts)
            parts = list(parts)
        else:
            count_parts = 0
            parts = []
        free_parts = 1 - sum(parts)
        free_slots = count_datasets - count_parts
        assert free_parts >= 0, (
            f"parts must sum up to 1.0, got = {free_parts} ({parts})!"
        )
        if free_slots > 0:
            # Fill the rest slots with part ratio up to 'count_datasets'
            parts.extend([free_parts / free_slots] * free_slots)
        self.datasets = [
            {
                'source': dataset,
                'index': 0,  # sequential access; TODO: random access
                'size': len(dataset),
                'part': part,
                'max': len(dataset) / part
            } for dataset, part in zip(datasets, parts)
        ]
        self.max = round(max([d['max'] for d in self.datasets]))
        # Indices of datasets like
        # [[0, 0, 0, ...], [1, 1, ...], ..., [n, n, n, n, ...]]
        # for n datasets
        indices = [[i] * round(self.max * d['part'])
                   for i, d in enumerate(self.datasets)]
        # print(indices)  # debug
        # Reduce [[0, 0, 0, ...], [1, 1, ...], ..., [n, n, n, n, ...]] ->
        # [0, 0, 0, ..., 1, 1, ..., n, n, n, n, ...]
        self.index = reduce(
            lambda x, y: x + y, indices, []
        )
        self.size = len(self.index)

    def __getitem__(self, i):
        item = self.datasets[self.index[i]]
        result = item['source'][item['index']]
        item['index'] = (item['index'] + 1) % item['size']
        return result

    def __len__(self):
        return self.size

    def reset(self):
        for i in range(len(self.datasets)):
            self.datasets[i]['index'] = 0
        return None

