# pylint: disable=C0111, R0913
"""
Elastic grid deformation initially based on procedure described in: Simard,
Patrice Y., David Steinkraus, and John C. Platt.  "Best practices for
convolutional neural networks applied to visual document analysis." Icdar. Vol.
3. No. 2003. 2003.

And then modified as described in:
https://arxiv.org/abs/1902.11050

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

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import im_utils


def get_indices(im_shape, scale, sigma, padding=60):
    """ based on cognitivemedium.com/assets/rmnist/Simard.pdf """
    im_shape = [im_shape[0] + (padding * 2), im_shape[1] + (padding * 2)]
    randx = np.random.uniform(low=-1.0, high=1.0, size=im_shape[:2])
    randy = np.random.uniform(low=-1.0, high=1.0, size=im_shape[:2])
    x_filtered = gaussian_filter(randx, sigma, mode="reflect") * scale
    y_filtered = gaussian_filter(randy, sigma, mode="reflect") * scale
    x_coords, y_coords = np.mgrid[0:im_shape[0], 0:im_shape[1]]
    x_deformed = x_coords + x_filtered
    y_deformed = y_coords + y_filtered
    return x_deformed, y_deformed


def get_elastic_map(im_shape, scale, intensity):
    assert 0 <= scale <= 1
    assert 0 <= intensity <= 1
    min_alpha = 200
    max_alpha = 2500

    min_sigma = 15
    max_sigma = 60
    alpha = min_alpha + ((max_alpha-min_alpha) * scale)
    alpha *= intensity
    sigma = min_sigma + ((max_sigma-min_sigma) * scale)
    return get_indices(im_shape, scale=alpha, sigma=sigma)


def transform_photo(image, def_map, padding=60):
    indices = def_map
    image = np.array(image)
    image = im_utils.pad(image, padding, mode='reflect')
    out_r = map_coordinates(image[:, :, 0], indices, order=1)
    out_g = map_coordinates(image[:, :, 1], indices, order=1)
    out_b = map_coordinates(image[:, :, 2], indices, order=1)
    image[:, :, 0] = out_r
    image[:, :, 1] = out_g
    image[:, :, 2] = out_b
    image = image[padding:-padding, padding:-padding]
    return image


def transform_annotation(image, def_map, padding=60):
    image = np.array(image)
    image = im_utils.pad(image, padding, mode='reflect')
    if len(image.shape) > 2:
        image = image.reshape(image.shape[:2])
    shape = image.shape
    out = map_coordinates(image[:, :], def_map, order=1).reshape(shape)
    out = out[padding:-padding, padding:-padding]
    out = np.round(out).astype(np.int64)
    return out
