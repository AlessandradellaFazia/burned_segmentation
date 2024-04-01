from datetime import datetime
from pathlib import Path


from abc import ABC

import numpy as np
from torch.utils.data import Dataset, Sampler

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class IndexedBounds:
    index: int
    coords: Tuple[int, int, int, int]


def to_tuple(value: Union[int, tuple]) -> tuple:
    """Convert a value to a tuple."""
    if isinstance(value, int):
        return (value, value)


def compute_padding(
    shape: Tuple[int, int], tile_size: int, stride: Optional[int] = None
) -> Tuple[int, int]:
    """Compute the padding to use for a given shape and tile size."""
    stride = stride or tile_size
    width, height = shape
    pad_w = (width // stride + 1) * stride - width
    pad_h = (height // stride + 1) * stride - height
    return pad_w, pad_h


def pad_shape(shape: Tuple[int, int], padding: Union[int, tuple]) -> Tuple[int, int]:
    """Pad a shape with a given padding."""
    width, height = shape
    pad_w, pad_h = to_tuple(padding)
    new_width = width + pad_w
    new_height = height + pad_h
    return new_width, new_height


class FullImageSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
    ) -> None:
        assert hasattr(
            dataset, "image_shapes"
        ), "To use samplers, the dataset must have an `image_shapes` method implementation."
        self.dataset = dataset
        self.shapes = dataset.image_shapes()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, image_size in enumerate(self.shapes):
            yield IndexedBounds(i, (0, 0, *image_size))


class TiledSampler(Sampler, ABC):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        length: int = None,
    ):
        assert hasattr(
            dataset, "image_shapes"
        ), "To use samplers, the dataset must have an `image_shapes` method implementation."
        self.dataset = dataset
        self.shapes = dataset.image_shapes()
        self.tile_size = tile_size
        self.length = length

    def __len__(self):
        if self.length:
            return self.length
        total = 0
        for w, h in self.shapes:
            num_horizontal = w // self.tile_size + int(w % self.tile_size != 0)
            num_vertical = h // self.tile_size + int(h % self.tile_size != 0)
            total += num_horizontal * num_vertical
        self.length = total
        return total


class SequentialTiledSampler(TiledSampler):
    def __iter__(self):
        for i, image_size in enumerate(self.shapes):
            width, height = image_size
            num_horizontal = width // self.tile_size + int(width % self.tile_size != 0)
            num_vertical = height // self.tile_size + int(height % self.tile_size != 0)

            for j in range(num_horizontal):
                for k in range(num_vertical):
                    x = j * self.tile_size
                    y = k * self.tile_size
                    yield IndexedBounds(i, (x, y, self.tile_size, self.tile_size))


class RandomTiledSampler(TiledSampler):
    def __init__(
        self,
        dataset: Dataset,
        tile_size: int,
        length: int = None,
    ):
        super().__init__(dataset, tile_size, length=length)
        self.indices = np.random.choice(len(self.dataset), len(self), replace=True)

    def __iter__(self):
        # sample a random image, then sample a random tile from that image
        for i in self.indices:
            width, height = self.shapes[i]
            x = np.random.randint(0, width - self.tile_size + 1, 1)
            y = np.random.randint(0, height - self.tile_size + 1, 1)
            yield IndexedBounds(i, (x, y, self.tile_size, self.tile_size))
