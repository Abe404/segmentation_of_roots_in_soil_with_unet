# pylint: disable=C0111,E1102,C0103
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


from math import ceil
import numpy as np
import skimage.util as skim_util
from skimage.exposure import rescale_intensity


def scale_zero_one(image):
    return rescale_intensity(image.astype(np.float32), out_range=(0, 1))


def pad(image, width: int, mode='reflect', constant_values=0):
    # only pad the first two dimensions
    pad_width = [(width, width), (width, width)]
    if len(image.shape) == 3:
        # don't pad channels
        pad_width.append((0, 0))
    if mode == 'reflect':
        return skim_util.pad(image, pad_width, mode)
    return skim_util.pad(image, pad_width, mode=mode,
                         constant_values=constant_values)


def add_salt_pepper(image, intensity):
    image = np.array(image)
    white = [1, 1, 1]
    black = [0, 0, 0]
    if len(image.shape) == 2 or image.shape[-1] == 1:
        white = 1
        black = 0
    num = np.ceil(intensity * image.size).astype(np.int)
    x_coords = np.floor(np.random.rand(num) * image.shape[1])
    x_coords = x_coords.astype(np.int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(np.int)
    image[x_coords, y_coords] = white
    x_coords = np.floor(np.random.rand(num) * image.shape[1]).astype(np.int)
    y_coords = np.floor(np.random.rand(num) * image.shape[0]).astype(np.int)
    image[y_coords, x_coords] = black
    return image


def add_gaussian_noise(image, sigma):
    assert np.min(image) >= 0
    assert np.max(image) <= 1
    gaussian_noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
    gaussian_noise = gaussian_noise.reshape(image.shape)
    return image + gaussian_noise


def get_train_tiles_and_masks(photos, annotations):
    all_tiles: list = []
    all_masks: list = []
    for photo, annotation in zip(photos, annotations):
        tiles, masks = get_random_tiles(photo,
                                        annotation, num_tiles=90)
        (tiles_with_roots,
         masks_with_roots) = get_tiles_and_masks_with_roots(tiles, masks)
        all_tiles += tiles_with_roots[:40]
        all_masks += masks_with_roots[:40]

    return all_tiles, all_masks


def get_val_tiles_and_masks(photos, annotations, in_tile_shape, out_tile_shape):
    all_tiles: list = []
    all_masks: list = []
    for photo, annotation in zip(photos, annotations):
        tiles, coords = get_tiles(photo, in_tile_shape, out_tile_shape)
        masks = tiles_from_coords(annotation, coords, out_tile_shape)
        all_tiles += tiles
        all_masks += masks
    return all_tiles, all_masks


def get_tiles(image, in_tile_shape, out_tile_shape):
    width_diff = in_tile_shape[1] - out_tile_shape[1]
    pad_width = width_diff // 2
    padded_photo = pad(image, pad_width)

    horizontal_count = ceil(image.shape[1] / out_tile_shape[1])
    vertical_count = ceil(image.shape[0] / out_tile_shape[0])

    # first split the image based on the tiles that fit
    x_coords = [h*out_tile_shape[1] for h in range(horizontal_count-1)]
    y_coords = [v*out_tile_shape[0] for v in range(vertical_count-1)]

    # The last row and column of tiles might not fit
    # (Might go outside the image)
    # so get the tile positiion by subtracting tile size from the
    # edge of the image.
    right_x = padded_photo.shape[1] - in_tile_shape[1]
    bottom_y = padded_photo.shape[0] - in_tile_shape[0]

    y_coords.append(bottom_y)
    x_coords.append(right_x)

    # because its a rectangle get all combinations of x and y
    tile_coords = [(x, y) for x in x_coords for y in y_coords]
    tiles = tiles_from_coords(padded_photo, tile_coords, in_tile_shape)
    return tiles, tile_coords


def reconstruct_from_tiles(tiles, coords, output_shape):
    image = np.zeros(output_shape)
    for tile, (x, y) in zip(tiles, coords):
        image[y:y+tile.shape[0], x:x+tile.shape[1]] = tile
    return image


def tiles_from_coords(image, coords, tile_shape):
    tiles = []
    for x, y in coords:
        tile = image[y:y+tile_shape[0],
                     x:x+tile_shape[1]]
        tiles.append(tile)
    return tiles


def get_random_tiles(image, annot, num_tiles, pad_width=92):
    tile_shape = (572, 572, 3)
    padded_photo = pad(image, pad_width)
    padded_annot = pad(annot, pad_width)
    right_limit = padded_photo.shape[1] - 572
    bottom_limit = padded_photo.shape[0] - 572
    x_coords = np.round(np.random.rand(num_tiles) * right_limit).astype(np.int)
    y_coords = np.round(np.random.rand(num_tiles) * bottom_limit).astype(np.int)
    tile_coords = list(zip(x_coords, y_coords))
    im_tiles = tiles_from_coords(padded_photo, tile_coords, tile_shape)
    annot_tiles = tiles_from_coords(padded_annot, tile_coords, (572, 572, 1))
    return im_tiles, annot_tiles


def get_tiles_and_masks_with_roots(tiles, masks):
    masks_to_keep = []
    tiles_to_keep = []
    for mask, tile in zip(masks, tiles):
        # make sure the centre region to be segmented has roots
        if np.sum(mask[92:-92, 92:-92]) > 0:
            masks_to_keep.append(mask)
            tiles_to_keep.append(tile)
    return tiles_to_keep, masks_to_keep
