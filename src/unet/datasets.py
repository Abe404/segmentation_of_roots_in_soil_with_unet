# pylint: disable=C0111, R0913, R0903
"""
Copyright (C) 2019 Abraham Smith

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from im_utils import scale_zero_one
from im_utils import get_train_tiles_and_masks
from im_utils import get_val_tiles_and_masks
import im_utils
import elastic

# pylint: disable=C0413
sys.path.append('.')
from data_utils import load_images_pool
from data_utils import load_annotations_pool


class UNetTransformer():
    def __init__(self):
        self.color_jit = transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                                saturation=0.2, hue=0.001)

    def transform(self, tile, mask):
        tile = im_utils.scale_zero_one(tile)
        assert np.sum(mask[92:-92, 92:-92]) > 0
        if random.random() < 0.90:
            def_map = elastic.get_elastic_map(tile.shape, scale=random.random(),
                                              intensity=0.4 + (0.6 * random.random()))
            tile = elastic.transform_photo(tile, def_map)
            mask = elastic.transform_annotation(mask, def_map)

        sigma = np.abs(np.random.normal(0, scale=0.09))
        tile = im_utils.add_gaussian_noise(tile, sigma)
        tile = im_utils.add_salt_pepper(tile, intensity=np.abs(np.random.normal(0.0, 0.008)))

        if random.random() < 0.5:
            tile = np.fliplr(tile)
            mask = np.fliplr(mask)

        tile -= np.min(tile)
        tile = np.divide(tile, np.max(tile)) * 255.0
        tile = Image.fromarray((tile).astype(np.int8), mode='RGB')
        tile = self.color_jit(tile)  # returns PIL image
        tile = np.array(tile)  # return back to numpy
        mask[mask > 0] = 1
        return tile, mask


class UNetTrainDataset(Dataset):
    def __init__(self, annotation_paths, photo_paths):
        self.augmentor = UNetTransformer()
        self.photos = load_images_pool(photo_paths)
        self.annotations = load_annotations_pool(annotation_paths)
        self.tile_shape = (572, 572, 3)
        self.assign_new_tiles()
        self.save_idx = 0

    def assign_new_tiles(self):
        self.tiles: list = []
        self.masks: list = []
        tiles, masks = get_train_tiles_and_masks(self.photos, self.annotations)
        assert masks and len(tiles) == len(masks)
        self.tiles = tiles
        self.masks = masks

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = np.array(self.tiles[idx])
        mask = np.array(self.masks[idx])
        tile, mask = self.augmentor.transform(tile, mask)
        tile = scale_zero_one(tile)
        mask = mask[92:-92, 92:-92]
        tile -= 0.5
        tile = np.moveaxis(tile, -1, 0)
        tile = tile.astype(np.float32)
        mask[mask > 0] = 1
        mask = mask.astype(np.int64)
        mask = mask.reshape(388, 388)
        tile = torch.from_numpy(tile)
        mask = torch.from_numpy(mask)
        return (tile, mask)


class UNetValDataset(Dataset):
    def __init__(self, annotation_paths, photo_paths):
        self.photos = load_images_pool(photo_paths)
        self.annotations = load_annotations_pool(annotation_paths)
        self.in_tile_shape = (572, 572, 3)
        self.out_tile_shape = (388, 388, 1)
        self.assign_new_tiles()
        self.save_idx = 0

    def assign_new_tiles(self):
        self.tiles: list = []
        self.masks: list = []
        tiles, masks = get_val_tiles_and_masks(self.photos, self.annotations,
                                               self.in_tile_shape, self.out_tile_shape)
        assert tiles and len(tiles) == len(masks)
        self.tiles = tiles
        self.masks = masks

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile = np.array(self.tiles[idx])
        mask = np.array(self.masks[idx])
        tile = scale_zero_one(tile)
        tile -= 0.5
        tile = np.moveaxis(tile, -1, 0)
        tile = tile.astype(np.float32)
        mask[mask > 0] = 1
        mask = mask.astype(np.int64)
        mask = mask.reshape(388, 388)
        tile = torch.from_numpy(tile)
        mask = torch.from_numpy(mask)
        return (tile, mask)
